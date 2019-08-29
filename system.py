import argparse
import json
import os
import math
import unicodedata

from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter
from tqdm import tqdm

from common.util.log_helper import LogHelper
from retrieval.top_n import TopNDocsTopNSents
from retrieval.fever_doc_db import FeverDocDB

import joblib
import tensorflow as tf
import random
import numpy as np
from fever.api.web_server import fever_web_api

from opt import adam, warmup_cosine, warmup_linear, warmup_constant
from text_utils import TextEncoder
from utils import encode_dataset, iter_data, find_trainable_variables, get_ema_vars, convert_gradient_to_tensor, shape_list, assign_to_gpu

def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

def swish(x):
    return x*tf.nn.sigmoid(x)

opt_fns = {
    'adam':adam,
}

act_fns = {
    'relu':tf.nn.relu,
    'swish':swish,
    'gelu':gelu
}

lr_schedules = {
    'warmup_cosine':warmup_cosine,
    'warmup_linear':warmup_linear,
    'warmup_constant':warmup_constant,
}


argmax = lambda x:np.argmax(x, 1)

pred_fns = {
    'rocstories':argmax,
    'entailment':argmax,
}

filenames = {
    'rocstories':'ROCStories.tsv',
    'entailment':'entailment.tsv',
}

label_decoders = {
    'rocstories':None,
    'entailment':None,
}

def _norm(x, g=None, b=None, e=1e-5, axis=[1]):
    u = tf.reduce_mean(x, axis=axis, keep_dims=True)
    s = tf.reduce_mean(tf.square(x-u), axis=axis, keep_dims=True)
    x = (x - u) * tf.rsqrt(s + e)
    if g is not None and b is not None:
        x = x*g + b
    return x

def norm(x, scope, axis=[-1]):
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        g, b = get_ema_vars(g, b)
        return _norm(x, g, b, axis=axis)

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1-pdrop)
    return x

def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)
    return w

def _attn(q, k, v, train=False, scale=False):
    w = tf.matmul(q, k)

    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))

    w = mask_attn_weights(w)
    w = tf.nn.softmax(w)

    w = dropout(w, attn_pdrop, train)

    a = tf.matmul(w, v)
    return a

def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1]+[n, m//n]
    return tf.reshape(x, new_x_shape)

def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)

def split_heads(x, n, k=False):
    if k:
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])

def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))

def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), pad='VALID', train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        if rf == 1: #faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
        else: #was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c

def attn(x, scope, n_state, n_head, train=False, scale=False):
    assert n_state%n_head==0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, 1, train=train)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        a = _attn(q, k, v, train=train, scale=scale)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, 1, train=train)
        a = dropout(a, resid_pdrop, train)
        return a

def mlp(x, scope, n_state, train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        act = act_fns[afn]
        h = act(conv1d(x, 'c_fc', n_state, 1, train=train))
        h2 = conv1d(h, 'c_proj', nx, 1, train=train)
        h2 = dropout(h2, resid_pdrop, train)
        return h2

def block(x, scope, train=False, scale=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, 'attn', nx, n_head, train=train, scale=scale)
        n = norm(x+a, 'ln_1')
        m = mlp(n, 'mlp', nx*4, train=train)
        h = norm(n+m, 'ln_2')
        return h

def embed(X, we):
    we = convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    h = tf.reduce_sum(e, 2)
    return h

def clf(x, ny, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), train=False):
    with tf.variable_scope('clf'):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w)+b

def model(X, M, Y, train=False, reuse=False):
    with tf.variable_scope('model', reuse=reuse):
        we = tf.get_variable("we", [n_vocab+n_special+n_ctx, n_embd], initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, embd_pdrop, train)

        X = tf.reshape(X, [-1, n_ctx, 2])
        M = tf.reshape(M, [-1, n_ctx])

        h = embed(X, we)
        for layer in range(n_layer):
            h = block(h, 'h%d'%layer, train=train, scale=True)

        lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(X[:, 1:, 0], [-1]))
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
        lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)

        clf_h = tf.reshape(h, [-1, n_embd])
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32)*n_ctx+pool_idx)

        # Reshape just so the dropout broadcasts evenly over examples
        if dataset == 'rocstories':
            clf_h = tf.reshape(clf_h, [-1, 2, n_embd])
        elif dataset == 'entailment':
            clf_h = tf.reshape(clf_h, [-1, 1, n_embd])
        if train and clf_pdrop > 0:
            shape = shape_list(clf_h)
            shape[1] = 1
            clf_h = tf.nn.dropout(clf_h, 1-clf_pdrop, shape)

        clf_h = tf.reshape(clf_h, [-1, n_embd])
        if dataset == 'rocstories':
            clf_logits = clf(clf_h, 1, train=train)
            clf_logits = tf.reshape(clf_logits, [-1, 2])
        elif dataset == 'entailment':
            clf_logits = clf(clf_h, 3, train=train)  # no reshape needed

        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)
        return clf_logits, clf_losses, lm_losses

def mgpu_predict(*xs):
    gpu_ops = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)
    for i, xs in enumerate(zip(*xs)):
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=True):
            clf_logits, clf_losses, lm_losses = model(*xs, train=False, reuse=True)
            gpu_ops.append([clf_logits, clf_losses, lm_losses])
    ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
    return ops

def transform_roc(X1, X2, X3):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 2, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        x12 = [start]+x1[:max_len]+[delimiter]+x2[:max_len]+[clf_token]
        x13 = [start]+x1[:max_len]+[delimiter]+x3[:max_len]+[clf_token]
        l12 = len(x12)
        l13 = len(x13)
        xmb[i, 0, :l12, 0] = x12
        xmb[i, 1, :l13, 0] = x13
        mmb[i, 0, :l12] = 1
        mmb[i, 1, :l13] = 1
    xmb[:, :, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
    return xmb, mmb

def transform_entailment(X1, X2):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 1, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 1, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2), in enumerate(zip(X1, X2)):
        x12 = [start]+x1[:max_len]+[delimiter]+x2[:max_len]+[clf_token]
        l12 = len(x12)
        xmb[i, 0, :l12, 0] = x12
        mmb[i, 0, :l12] = 1
    xmb[:, :, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
    return xmb, mmb

def iter_predict(Xs, Ms):
    logits = []
    print("Predicting")
    for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
        n = len(xmb)
        if n == n_batch_train:
            logits.append(sess.run(eval_mgpu_logits, {X_train:xmb, M_train:mmb}))
        else:
            logits.append(sess.run(eval_logits, {X:xmb, M:mmb}))
    logits = np.concatenate(logits, 0)
    return logits



def process_line(method, line):
    sents = method.get_sentences_for_claim(line["claim"])
    pages = list(set(map(lambda sent:sent[0],sents)))
    line["predicted_pages"] = pages
    line["predicted_sentences"] = sents
    return line

def resolve_evidence(sents):
    found_evidence = []

    for found_sentence in sents:
        title = unicodedata.normalize('NFD', str(found_sentence[0]))
        linenum = found_sentence[1]
        doc = db.get_doc_lines(title)

        line = doc.split("\n")[linenum]
        sentence_text = line.split("\t")

        label = "UNCLASSIFIED"

        found_evidence.append({"title": title,
                                           "line_number": linenum,
                                           "text": sentence_text[1],
                                           "label": label})

    return found_evidence

def make_instances(master, evidence):
    global max_sent_len
    instances = []

    for idx, evi in enumerate(evidence):
        underlined_title = evi["title"]
        label = 0
        premise = evi["text"]

        # Prefix the premise sentence with [ TITLE ] (from source article)
        title = underlined_title.replace("_", " ")
        title_words = tokenizer.split_words(title)
        tokenized_title = " ".join(map(lambda x: x.text, title_words))
        premise = "[ " + tokenized_title + " ] " + premise

        premise_words = premise.split(" ")
        if len(premise_words) > max_sent_len:
            premise = " ".join(premise_words[0:max_sent_len])

        hypothesis = master["tokenized_claim"]
        instances.append({"index":idx,
                          "id":master["id"] if "id" in master else 0,
                          "premise": premise,
                          "hypothesis":hypothesis,
                          "label": label
        })




    return instances

def predict_sub_instances(text_encoder, sub_instances):
    global dataset
    if not len(sub_instances):
        return []
    prems, hyps, ys = zip(*[(sub["premise"], sub["hypothesis"], sub["label"]) for sub in sub_instances])
    test_set = encode_dataset([(prems, hyps, ys)], encoder=text_encoder)
    (tst_p, tst_h, teY) = test_set[0]
    teX, teM = transform_entailment(tst_p, tst_h)
    pred_fn = pred_fns[dataset]
    label_decoder = label_decoders[dataset]

    predictions = pred_fn(iter_predict(teX, teM))
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]

    return predictions

def fever_app(caller):


    global db, tokenizer, text_encoder, encoder, X_train, M_train, X, M, Y_train, Y,params,sess, n_batch_train, db_file, \
        drqa_index, max_page, max_sent, encoder_path, bpe_path, n_ctx, n_batch, model_file
    global n_vocab,n_special,n_y,max_len,clf_token,eval_lm_losses,eval_clf_losses,eval_mgpu_clf_losses,eval_logits, \
        eval_mgpu_logits,eval_logits

    LogHelper.setup()
    logger = LogHelper.get_logger("papelo")

    logger.info("Load config")
    config = json.load(open(os.getenv("CONFIG_FILE","configs/config-docker.json")))
    globals().update(config)
    print(globals())

    logger.info("Set Seeds")
    random.seed(42)
    np.random.seed(42)
    tf.set_random_seed(42)

    logger.info("Load FEVER DB")
    db = FeverDocDB(db_file)
    retrieval = TopNDocsTopNSents(db, max_page, max_sent, True, False, drqa_index)

    logger.info("Init word tokenizer")
    tokenizer = SimpleWordSplitter()

    # Prepare text encoder
    logger.info("Load BPE Text Encoder")
    text_encoder = TextEncoder(encoder_path, bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    n_y = 3
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    n_special = 3
    max_len = n_ctx // 2 - 2

    n_batch_train = n_batch

    logger.info("Create TF Placeholders")
    X_train = tf.placeholder(tf.int32, [n_batch, 1, n_ctx, 2])
    M_train = tf.placeholder(tf.float32, [n_batch, 1, n_ctx])
    X = tf.placeholder(tf.int32, [None, 1, n_ctx, 2])
    M = tf.placeholder(tf.float32, [None, 1, n_ctx])

    Y_train = tf.placeholder(tf.int32, [n_batch])
    Y = tf.placeholder(tf.int32, [None])

    logger.info("Model Setup")
    eval_logits, eval_clf_losses, eval_lm_losses = model(X, M, Y, train=False, reuse=None)
    eval_mgpu_logits, eval_mgpu_clf_losses, eval_mgpu_lm_losses = mgpu_predict(X_train, M_train, Y_train)

    logger.info("Create TF Session")
    params = find_trainable_variables('model')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=float(os.getenv("TF_GPU_MEMORY_FRACTION","0.5")))
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    sess.run([p.assign(ip) for p, ip in zip(params, joblib.load(model_file))])

    logger.info("Ready")

    def predict(instances):
        predictions = []

        for instance in tqdm(instances):
            sents = retrieval.get_sentences_for_claim(instance["claim"])
            found_evidence = resolve_evidence(sents)
            instance["tokenized_claim"] = " ".join(map(lambda x: x.text, tokenizer.split_words(instance["claim"])))

            sub_instances = make_instances(instance, found_evidence)
            sub_predictions = predict_sub_instances(text_encoder, sub_instances)

            refute_evidence =  [i for i, x in enumerate(sub_predictions) if x == 2]
            support_evidence = [i for i, x in enumerate(sub_predictions) if x == 0]

            if len(support_evidence):
                predicted_label = "SUPPORTS"
                predicted_evidence = [[found_evidence[i]["title"], found_evidence[i]["line_number"]] for i in support_evidence]
            elif len(refute_evidence):
                predicted_label = "REFUTES"
                predicted_evidence = [[found_evidence[i]["title"], found_evidence[i]["line_number"]] for i in refute_evidence]
            else:
                predicted_label = "NOT ENOUGH INFO"
                predicted_evidence = []

            predictions.append({"predicted_label":predicted_label,
                                "predicted_evidence": predicted_evidence})




        return predictions

    return caller(predict)

def web():
    return fever_app(fever_web_api)


if __name__ == "__main__":
    call_method = None

    def cli_method(predict_function):
        global call_method
        call_method = predict_function

    def cli():
        return fever_app(cli_method)

    cli()

    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file")
    parser.add_argument("--out-file")
    args = parser.parse_args()

    claims = []

    with open(args.in_file,"r") as in_file:
        for text_line in in_file:
            line = json.loads(text_line)
            claims.append(line)

    ret = call_method(claims)

    with open(args.out_file,"w+") as out_file:
        for prediction in ret:
            out_file.write(json.dumps(prediction)+"\n")

