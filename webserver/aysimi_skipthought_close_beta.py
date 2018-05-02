#!/usr/bin/env python
# -*- coding:utf-8 -*-

import ConfigParser
import json
import urllib

import tornado.ioloop
import tornado.web
import os
import sys
import re
import time
import urllib2

import node
import tensorflow as tf
from skip_thought_model import SkipThoughtModel
from prodata.data_utils import TextData
from segcont.seg_cn_a_word import SegCNAWord
from log_factory import FinalLogger

if sys.getdefaultencoding() is not 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

re_sub = re.compile(r'\s')

# similar anyou server log
simi_log = None

# work data
work_data_dir = '../data/'
work_seed_file = 'seed.txt'
work_seed_mat = 'seed.infer'
work_simi_file = 'simi.txt'
work_simi_mat = 'simi.mat'

thread_num = 64
word_vec_dim = 100
server_port = '10107'
class_server_url = 'http://172.16.124.16:8398/anyouClassify'

tf_sess = None
tf_gragh = None
text_data = None
seg_a_word = None
skip_thought_model = None

# minshi json data
minshi_firstlist = list()
minshi_nodemap = dict()
minshi_label_map = dict()
# xingshi json data
xingshi_firstlist = list()
xingshi_nodemap = dict()
xingshi_label_map = dict()


tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
tf.app.flags.DEFINE_integer('pred_batch_size', 1, 'Predict batch Size.')
tf.app.flags.DEFINE_string('unit_type', 'gru', 'lstm | gru')
tf.app.flags.DEFINE_integer('num_units', 100, 'RNN Size.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in the model.')
tf.app.flags.DEFINE_float('dropout', 0.0, 'Dropout rate (not keep_prob)')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'Encoder & Decoder embedding size.')
tf.app.flags.DEFINE_integer('num_keep_ckpts', 10, 'Max number of checkpoints to keep.')
tf.app.flags.DEFINE_integer('target_max_len', 100, 'Target sentence max length.')
tf.app.flags.DEFINE_integer('max_vocab_size', 20000, 'Max vocab size.')
tf.app.flags.DEFINE_string('checkpoint_dir', '../model/mix/', 'Path to model.')
tf.app.flags.DEFINE_string('train_data_path', '../data/mix.txt.a.word', 'Path to file with train data.')

FLAGS = tf.app.flags.FLAGS


# Usage java -jar topNSim.jar num_thread vec_dim top_num src_file tgt_file
def aysimi_topN_java(num_thread, vec_dim, top_num, src_file, tgt_file):
    topN_cmd = \
        'java -jar topNSim.jar ' + str(num_thread) + ' ' + str(vec_dim) + ' ' + \
        str(top_num) + ' ' + src_file + ' ' + tgt_file
    simi_log.info(topN_cmd)
    res_simi = os.popen(topN_cmd).readlines()
    simi_log.info('Similar result: %s.', str(res_simi))
    return res_simi


def get_anyou_type(seed):
    """anyou type"""
    if not seed or 'caseText' not in seed:
        return False

    anyou_dm = '0'
    if 'dm' in seed:
        anyou_dm = seed['dm']

    anyou_type = 'minshi'
    if anyou_dm in minshi_nodemap.keys():
        anyou_type = 'minshi'
    elif anyou_dm in xingshi_nodemap.keys():
        anyou_type = 'xingshi'
    else:
        req_dict = dict()
        req_dict['src'] = seed['caseText']
        req_encode = urllib.urlencode(req_dict)
        req_post = req_encode.encode('utf-8')
        req = urllib2.Request(url=class_server_url, data=req_post)
        res = urllib2.urlopen(req)
        res = res.read().decode('utf-8')
        res = json.loads(res)

        if 'minshi' in res.keys() and 'labels' in res['minshi'].keys() and 'prob' in res['minshi'].keys() \
                and 'status' in res['minshi']['labels'] and res['minshi']['labels']['status'] == 'ok':
            anyou_type = 'minshi'
        else:
            anyou_type = 'xingshi'

    if not re.match(r'^\d+$', anyou_dm):
        anyou_dm = '0'

    return anyou_dm, anyou_type


def init_tf_model():
    global tf_sess
    global tf_gragh
    global text_data
    global skip_thought_model

    simi_log.info('Init skip_thought model.')
    tf_gragh = tf.Graph()
    with tf_gragh.as_default():

        text_data = TextData(
            FLAGS.train_data_path, max_vocab_size=FLAGS.max_vocab_size, max_len=FLAGS.target_max_len)

        skip_thought_model = SkipThoughtModel(
            len(text_data.vocab), text_data.vocab.START_VOCAB, FLAGS.target_max_len,
            FLAGS.unit_type, FLAGS.num_units, FLAGS.num_layers, FLAGS.dropout,
            FLAGS.embedding_size, FLAGS.learning_rate, FLAGS.num_keep_ckpts)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.allow_growth = True
    tf_sess = tf.Session(graph=tf_gragh, config=config)
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        skip_thought_model.saver.restore(tf_sess, ckpt.model_checkpoint_path)
    else:
        pass


def stop_tf_model():
    global tf_sess

    if tf_sess:
        tf_sess.close()


def infer_tf_model(seed, simi_list):
    global tf_sess
    global text_data
    global skip_thought_model
    global seg_a_word

    if not seed or not simi_list:
        return

    simi_log.info('Begin: skip_thought model infer.')
    with open(work_data_dir + work_seed_file, 'w') as f:
        tmp_res = seg_a_word.seg_cont(re_sub.sub('', seed['caseText']))
        for tmp_i in tmp_res:
            f.write((tmp_i + ' ').encode('utf-8'))
    with open(work_data_dir + work_seed_mat, 'w') as f:
        pred_data = text_data.pro_tuple_data(work_data_dir + work_seed_file, batch_size=FLAGS.pred_batch_size)
        for j, batch in enumerate(pred_data):
            if batch == text_data.ONE_LINE_TOKEN:
                break
            prev_predict, curr_state = tf_sess.run(
                [skip_thought_model.prev_predict_logits, skip_thought_model.curr_encoder_state],
                feed_dict={skip_thought_model.curr_source_data: batch.data,
                           skip_thought_model.curr_source_seq_len: batch.seq_lengths}
            )

            # simi_log.info('%d, %s', j, '------')
            # for pred_i in prev_predict:
            #     pred_str = ''
            #     for pred_j in pred_i:
            #         pred_str += text_data.vocab.index2word[pred_j] + ','
            #     simi_log.info(pred_str)

            f.write((' '.join(map(str, curr_state[-1][-1])) + ' ').encode('utf-8'))

    with open(work_data_dir + work_simi_mat, 'w') as f_mat:
        for simi_one in simi_list:
            with open(work_data_dir + work_simi_file, 'w') as f_simi:
                tmp_res = seg_a_word.seg_cont(re_sub.sub('', simi_one['case_newcontent']))
                for tmp_i in tmp_res:
                    f_simi.write((tmp_i + ' ').encode('utf-8'))

            # simi_log.info('\n------')
            pred_data = text_data.pro_tuple_data(work_data_dir + work_simi_file, batch_size=FLAGS.pred_batch_size)
            for j, batch in enumerate(pred_data):
                if batch == text_data.ONE_LINE_TOKEN:
                    break
                # simi_log.info('line_num: %s, %s : %s', str(j), str(batch.seq_lengths), str(batch.data))
                prev_predict, curr_state = tf_sess.run(
                    [skip_thought_model.prev_predict_logits, skip_thought_model.curr_encoder_state],
                    feed_dict={skip_thought_model.curr_source_data: batch.data,
                               skip_thought_model.curr_source_seq_len: batch.seq_lengths}
                )

                f_mat.write((' '.join(map(str, curr_state[-1][-1])) + ' ').encode('utf-8'))
            f_mat.write('\n')
    simi_log.info('End: skip_thought model infer.')


def get_simi_list(seed, simi_list):
    if not seed or not simi_list or 'caseText' not in seed:
        return False

    try:
        # anyou_dm, anyou_type = get_anyou_type(seed)
        # if not (anyou_dm and anyou_type):
        #     return False

        infer_tf_model(seed, simi_list)

        res_doc = aysimi_topN_java(
            thread_num, word_vec_dim, len(simi_list),
            work_data_dir + work_seed_mat, work_data_dir + work_simi_mat)

        seq_ay = dict()
        for i in range(len(res_doc)):
            doc_one = res_doc[i].split('\t')
            seq_ay[int(doc_one[1])] = i + 1

        for i in range(len(simi_list)):
            simi_list[i]['seq_no'] = seq_ay[i]
    except Exception, e:
        simi_log.info('%s, %s', Exception.__name__, e)
        return False

    return True


class SimiCloseBetaCaseHandler(tornado.web.RequestHandler):
    def get(self):
        pass

    def post(self):
        try:
            start_time = time.time()
            raw_data = self.request.body
            simi_log.info(self.request.remote_ip + ' raw:' + raw_data)
            # res = self.request.arguments
            res = json.loads(self.request.body[len('similarCaseList='):len(self.request.body) - 1], strict=False)

            # seed_dict = res['similarCaseList']['seeds']
            # similar_list = res['similarCaseList']['similar']
            seed_dict = res['seeds']
            similar_list = res['similar']
            status = get_simi_list(seed_dict, similar_list)
            res_order = dict()
            res_order['similar'] = similar_list
            res_order['status'] = str(status)
            self.write(json.dumps(res_order))
            elapse_time = time.time() - start_time
            simi_log.info('Elapse time: %s, status: %s.', str(elapse_time), str(status))
        except Exception, e:
            simi_log.info('%s, %s', Exception.__name__, e)
            self.write(e)


def main():
    global simi_log
    global work_data_dir
    global thread_num
    global word_vec_dim
    global class_server_url
    global seg_a_word

    global minshi_firstlist
    global minshi_nodemap
    global minshi_label_map
    global xingshi_firstlist
    global xingshi_nodemap
    global xingshi_label_map

    simi_log = FinalLogger('aysimi_skipthought_close_beta.log')

    seg_a_word = SegCNAWord()

    cp = ConfigParser.SafeConfigParser()
    cp.read('conf_aysimi_skipthought_close.conf')

    server_port = cp.get('server', 'port')

    work_data_dir = cp.get('simi_calc', 'work_data_dir')
    thread_num = cp.get('simi_calc', 'thread_num')
    word_vec_dim = FLAGS.num_units

    class_server_url = cp.get('class_server', 'server_url')

    node.loadConfig(work_data_dir + 'AY_minshi.xml', minshi_firstlist, minshi_nodemap, minshi_label_map)
    node.loadConfig(work_data_dir + 'AY_xingshi.xml', xingshi_firstlist, xingshi_nodemap, xingshi_label_map)

    init_tf_model()

    simi_log.info('---anyou similar skipthought close beta start server---')

    application.listen(server_port)
    tornado.ioloop.IOLoop.instance().start()


settings = {
    'static_path': os.path.join(os.path.dirname(__file__), 'anyou'),
    'cookie_secret': '61oETzKXQAGaYdkL5gEmGeJJFuYh7EQnp2XdTP1o/Vo=',
    'login_url': '/login',
    'xsrf_cookies': False,
}

application = tornado.web.Application([
    (r'/simiCloseBetaCase', SimiCloseBetaCaseHandler)
], **settings)

if __name__ == '__main__':
    main()
