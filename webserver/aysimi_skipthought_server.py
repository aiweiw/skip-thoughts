#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import

import ConfigParser
import codecs

import tornado.ioloop
import tornado.web
import os
import jieba
import sys
import re
import time
import tensorflow as tf

from skip_thought_model import SkipThoughtModel
from prodata.data_utils import TextData
from segcont.seg_cn_a_word import SegCNAWord
from log_factory import FinalLogger

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
tf.app.flags.DEFINE_integer('pred_batch_size', 1, 'Predict batch Size.')
tf.app.flags.DEFINE_string('unit_type', 'gru', 'lstm | gru')
tf.app.flags.DEFINE_integer('num_units', 100, 'RNN Size.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in the model.')
tf.app.flags.DEFINE_float('dropout', 0.0, 'Dropout rate (not keep_prob)')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'Encoder & Decoder embedding size.')
tf.app.flags.DEFINE_integer('num_keep_ckpts', 5, 'Max number of checkpoints to keep.')
tf.app.flags.DEFINE_integer('target_max_len', 50, 'Target sentence max length.')
tf.app.flags.DEFINE_integer('max_vocab_size', 20000, 'Max vocab size.')
tf.app.flags.DEFINE_string('checkpoint_dir', '../model/', 'Path to model.')
tf.app.flags.DEFINE_string('train_data_path', '../data/9107.test.train', 'Path to file with train data.')
tf.app.flags.DEFINE_string('pred_src_path', '../data/9107.test', 'Path to file of predict source data.')
tf.app.flags.DEFINE_string('pred_tgt_path', '../data/9107.test.matrix', 'Path to file of predict matrix data.')

FLAGS = tf.app.flags.FLAGS

if sys.getdefaultencoding() is not 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

server_port = ''  # similar anyou server port
simi_log = None  # similar anyou server log

thread_num = 64
word_vec_dim = 100
# top_num_simi = 50
work_data_dir = '../data/'
work_src_file = 'aytrain.txt'
work_src_matrix = 'aytrain.txt.matrix'
work_test_matrix = 'aytrain.test.matrix'

tf_sess = None
tf_gragh = None
text_data = None
seg_a_word = None
skip_thought_model = None


# Usage java -jar topNSim.jar num_thread vec_dim top_num src_file tgt_file
def aysimi_topN_java(num_thread, vec_dim, top_num, src_file, tgt_file):
    topN_cmd = \
        'java -jar topNSim.jar ' + str(num_thread) + ' ' + str(vec_dim) + ' ' + \
        str(top_num) + ' ' + src_file + ' ' + tgt_file
    res_simi = os.popen(topN_cmd).readlines()
    return res_simi


def parse_data(res_cmd=None):
    """
    :param res_cmd:
    :return:
    """
    global simi_log
    global work_src_file

    if not res_cmd:
        simi_log.info('Error: aysimi_topN_java.')
        return

    res_one_data_size = 4
    res_val = list()

    for res_i in res_cmd:
        doc_val = res_i.strip(' \r\n').split('\t')
        if len(doc_val) < res_one_data_size:
            continue

        val_one = dict()
        val_one['file'] = work_src_file  # doc_val[0]
        val_one['id'] = doc_val[1]
        val_one['simi'] = doc_val[2]

        list_row_col = []
        target_list = [x.strip(' ') for x in re.compile(r',').split(doc_val[3][1:-1].strip(' \n'))]
        for i in range(len(target_list)):
            one_row_col = [x for x in target_list[i].split(' ')]
            list_row_col.append((int(one_row_col[0]), int(one_row_col[1])))

        val_one['row_col'] = list_row_col

        res_val.append(val_one)

    return res_val


def insert_text(res_doc=None):
    """
    :param res_doc:
    :return:
    """
    global simi_log
    global work_data_dir
    global work_src_file

    if not res_doc:
        simi_log.info('No data: res_doc')
        return

    doc_loc = dict()
    for i in range(len(res_doc)):
        doc_loc[res_doc[i]['file'] + '-' + str(res_doc[i]['id'])] = i

    fp = None
    try:
        srcfile = work_data_dir + work_src_file
        fp = codecs.open(srcfile, 'r', 'utf-8')
        line = fp.readline()
        line_num = 0
        gotten = 0
        while line:
            doc_index = work_src_file + '-' + str(line_num)
            if doc_index in doc_loc.keys():
                res_doc[doc_loc[doc_index]]['src'] = line
                gotten += 1
                if gotten >= len(res_doc):
                    break
            line = fp.readline()
            line_num += 1
        fp.close()
    except Exception, e:
        simi_log.info('%s, %s', Exception.__name__, e)
    finally:
        if fp and not fp.closed:
            fp.close()


def clean_text(res_doc=None):
    """
    :param res_doc:
    :return:
    """
    global simi_log

    if not res_doc:
        return

    try:
        # remove no data's res
        res_del = list()
        for res_one in res_doc:
            if 'src' not in res_one:
                res_del.append(res_one)
        for i_del in res_del:
            res_doc.remove(i_del)

        res_cont = list()
        for res_one in res_doc:
            res_cont.append(res_one['src'])

        res_k = list()
        res_v = list()
        for cont_i in res_cont:
            kv = [x.strip(' \n\r') for x in cont_i.split('\t')]
            res_k.append(kv[0])
            res_v.append(kv[1])

        from collections import defaultdict
        res_k_index = defaultdict(list)
        for v, i in [(v, i) for i, v in enumerate(res_k)]:
            res_k_index[v].append(i)

        res_del = list()
        for k, v in res_k_index.iteritems():
            if len(v) <= 1:
                continue
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    if res_v[v[i]] in res_v[v[j]]:
                        res_del.append(res_doc[v[i]])
                    elif res_v[v[j]] in res_v[v[i]]:
                        res_del.append(res_doc[v[j]])

        for i_del in res_del:
            res_doc.remove(i_del)

    except Exception, e:
        simi_log.info('%s, %s', Exception.__name__, e)


def get_aysimi_demo(src, top_num):
    """
    :param src:
    :param top_num:
    :return:
    """
    global simi_log
    global thread_num
    global word_vec_dim
    global work_data_dir
    global work_src_matrix
    global work_test_matrix

    start_time = time.time()

    file_src_matrix = work_data_dir + work_src_matrix
    file_test_matrix = work_data_dir + work_test_matrix
    if not os.path.exists(file_src_matrix):
        return

    infer_tf_model(src, file_test_matrix)

    # top_num = int(os.popen('wc -l ' + file_src_matrix).readlines()[0].split(' ')[0])
    res_topN = aysimi_topN_java(thread_num, word_vec_dim, top_num, file_test_matrix, file_src_matrix)

    res_doc = parse_data(res_topN)

    if not res_doc:
        return

    insert_text(res_doc)
    # clean_text(res_doc)

    def transcode(x):
        str_x = '{'
        str_x += "\'file\': \'" + x['file'] + '\', '
        str_x += "\'id\': \'" + x['id'] + '\', '
        str_x += "\'simi\': \'" + x['simi'] + '\', '
        str_x += "\'row_col\': " + str(x['row_col']) + ', '
        str_x += "\'src\': \'" + x['src'].strip(' \r\n').encode('utf-8') + '\'}'
        return str_x

    simi_log.info('[' + ', '.join(map(transcode, res_doc)) + ']')

    simi_log.info('Elapse time: ' + str((time.time() - start_time)))

    return res_doc


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write(
            "<html><head><title>案件分析调用接口</title></head><body>案件分析调用接口")
        self.write("&nbsp;&nbsp;&nbsp; <a href=/similarCaseDemo>api调用demo</a> &nbsp;&nbsp;&nbsp;")
        self.write("<br>方法：   similarCase ")
        self.write("<br>功能：   根据一段文本匹配相似案例 ")
        self.write("<br>调用方法：http请求   请求URL：  http://$host:$port/similarCase")
        self.write("<br>请求类型：post    ")
        self.write("<br>请求参数：序列化后的json对象，包括1个字段，必选字段：src(原文)")
        self.write("<br>编码支持：utf-8 ")
        self.write("<br>返回结果：Json对象序列化后的字符串")
        self.write("<br>")


class SimilarCaseDemoHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('<html><body><form action="/similarCaseDemo" method="post">'
                   '<p><b>相似案例查找demo</b></p>'
                   '<p>案件基本情况</p>'
                   '<textarea name="message" style="width:600px;height:300px;">'
                   """原告谭成亮诉称，原告系渝北区某街道某小区某幢某单元某号房屋所有权人，依法享有业主权利。2013年7月6日，被告突然宣布要更换小区物业服务企业，并声称“大修基金是国家的，不用白不用”等。原告认为被告的言行不符合法律规定，原告多次向被告及社区反映情况，要求公开业主大会、业委会的有关规则、制度，公开某小区至今的相关情况和问题，召开业主大会讨论决定如何解决小区的问题，至今没有得到正面答复。请求人民法院依法判决：被告公布、提供下列情况和资料，1、某小区业主大会议事规则、管理规约；2、某小区业主大会和被告从成立至今的决定及会议记录；3、某小区建筑物及其附属设施维修资金的筹集、使用情况；4、某小区共有部分的使用和收益情况；5、被告从成立至今的财务收支明细情况；6、被告的印章管理制度及使用详情。
                   """
                   '</textarea>'
                   '<p>&nbsp;</p>'
                   '<label><input name="if_classify" type="checkbox" value="identy"/>自动识别案由</label>'
                   '&nbsp;&nbsp;&nbsp;'
                   '<select name="anyou_type" style="width:100px;">'
                   '<option value="">自动识别</option>'
                   '<option value="minshi">民事</option>'
                   '<option value="xingshi">刑事</option>'
                   '</select>'
                   '<p>&nbsp;</p>'
                   '<label><input name="if_classify" type="checkbox" value="assign" />指定输入案由</label>'
                   '&nbsp;&nbsp;&nbsp;'
                   '<select name="correct_anyou_type" style="width:100px;">'
                   '<option value="minshi">民事</option>'
                   '<option value="xingshi">刑事</option>'
                   '</select>'
                   '&nbsp;&nbsp;&nbsp;'
                   '<input type="text" name="classify_id" />'
                   '(例: 9000，9001)'
                   '<br>'
                   '<a href=/anyou/xingshi_classify.json>刑事案由参考</a>&nbsp;&nbsp;&nbsp;'
                   '<a href=/anyou/minshi_classify.json>民事案由参考</a> </p>'
                   '<p>&nbsp;</p>'
                   '界面显示'
                   '&nbsp;&nbsp;&nbsp;'
                   '<label><input name="view_similar" type="radio" value="simi_word" checked="checked"/>案例匹配</label>'
                   '<label><input name="view_similar" type="radio" value="simi_line"/>短句匹配</label>'
                   '<br><br> <input type="submit" value="Submit">'
                   '</form>'
                   '</body></html>')

    def post(self):
        self.set_header("Content-Type", "text/html")
        src = self.get_argument("message")
        view_similar = str(self.get_argument("view_similar"))
        simi_log.info(self.request.remote_ip + ', similar demo: ' + src)
        data = dict()
        data["top"] = 10
        data["src"] = src
        result = get_aysimi_demo(data["src"], data["top"])

        stopword_set = set(
            ["年", "月", "日", "的", "了", "将", "诉称" , "后", "于", "并", "但", "与", "元", "万元" , "”", "、",
             "《", "》", "：", "；", "，", "。"]
        )
        src_word_list = jieba.cut(src, cut_all=False)
        src_word_set = set()
        digits = re.compile(r"\d+")
        for word in src_word_list:
            digit_match = re.match(digits, word)
            if digit_match:
                continue
            if word.encode() not in stopword_set:
                src_word_set.add(word)

        from matrixText.matrix_seg import seg_sentence
        src_list = seg_sentence(src)

        self.write('<!DOCTYPE html>'
                   '<html><head>'
                   '<meta http-equiv="content-type" content="text/html;charset=utf-8">'
                   '<title>相似案例结果</title>')
        self.write("</head>")
        self.write("<body>")
        self.write("<div>")

        self.write('<table border = "1">'
                   '<tr>'
                   '<th style="width: 40%">' + '源文' + '</th>'
                                                      '<th style="width: 60%">相似案例</th>'
                                                      '</tr>'
                   )

        separation = '<br>'
        i = 1
        for res_one in result:
            if "src" not in res_one.keys():
                continue

            if 'simi_line' == view_similar:

                self.write('<tr>')
                self.write('<td>')
                self.write("<p>" + str(i) + ":&nbsp;&nbsp;&nbsp;&nbsp;")
                self.write(separation)

                res_test_sym = list()
                res_norm_sym = list()
                for j in range(len(res_one['row_col'])):
                    res_test_sym.append(res_one['row_col'][j][0])
                    res_norm_sym.append(res_one['row_col'][j][1])

                for j in range(len(src_list)):
                    if j in res_test_sym:
                        if res_test_sym.index(j) == 0:
                            self.write("<b><font color=\"red\">" + '[ ' + str(j) + ' - ' + str(res_norm_sym[0])
                                       + ' ]' + '<sub>' + str(0) + '</sub>' + ': ' + src_list[j] + "</font></b>")
                            self.write(separation)
                        elif res_test_sym.index(j) in [1, 2, 3]:
                            self.write("<b><font color=\"blue\">" + '[ ' + str(j) + ' - ' + str(
                                res_norm_sym[res_test_sym.index(j)])
                                       + ' ]' + '<sub>' + str(res_test_sym.index(j)) + '</sub>' + ': '
                                       + src_list[j] + "</font></b>")
                            self.write(separation)
                        else:
                            self.write("<b><font color=\"olive\">" + '[ ' + str(j) + ' - ' + str(
                                res_norm_sym[res_test_sym.index(j)])
                                       + ' ]' + '<sub>' + str(res_test_sym.index(j)) + '</sub>' + ': ' + "</font></b>")
                            self.write(src_list[j])
                            self.write(separation)
                    elif src_list[j] and not re.compile(r'^\s*\n*$').match(src_list[j]):
                        self.write('[' + str(j) + ']: ' + src_list[j])
                        self.write(separation)
                self.write('</td>')

                self.write('<td>')
                self.write("<p>" + str(i) + ":&nbsp;&nbsp;&nbsp;&nbsp;")
                self.write(separation)
                doc_list = seg_sentence(res_one["src"])

                for j in range(len(doc_list)):
                    if j in res_norm_sym:
                        if res_norm_sym.index(j) == 0:
                            self.write("<b><font color=\"red\">" + '[ ' + str(j) + ' - ' + str(res_test_sym[0])
                                       + ' ]' + '<sub>' + str(0) + '</sub>' + ': ' + doc_list[j] + "</font></b>")
                            self.write(separation)
                        elif res_norm_sym.index(j) in [1, 2, 3]:
                            self.write(
                                "<b><font color=\"blue\">" + '[ ' + str(j) + ' - ' + str(
                                    res_test_sym[res_norm_sym.index(j)])
                                + ' ]' + '<sub>' + str(res_norm_sym.index(j)) + '</sub>' + ': '
                                + doc_list[j] + "</font></b>")
                            self.write(separation)
                        else:
                            self.write(
                                "<b><font color=\"olive\">" + '[ ' + str(j) + ' - ' + str(
                                    res_test_sym[res_norm_sym.index(j)])
                                + ' ]' + '<sub>' + str(res_norm_sym.index(j)) + '</sub>' + ': ' + "</font></b>")
                            self.write(doc_list[j])
                            self.write(separation)
                    elif doc_list[j] and not re.compile(r'^\s*\n*$').match(doc_list[j]):
                        self.write('[' + str(j) + ']: ' + doc_list[j])
                        self.write(separation)

                self.write('</td>')
                self.write('</tr>')

            elif 'simi_word' == view_similar or view_similar == []:

                self.write('<tr>')
                self.write('<td>')
                self.write("<p>" + str(i) + ":&nbsp;&nbsp;&nbsp;&nbsp;")
                self.write(separation)

                self.write(str(src))
                self.write('</td>')

                self.write('<td>')
                self.write("<p>" + str(i) + ":&nbsp;&nbsp;&nbsp;&nbsp;")
                self.write(separation)

                target_word_list = jieba.cut(res_one["src"], cut_all=False)
                sameCount = 0
                sameWords = ""
                for word in target_word_list:
                    if word in src_word_set:
                        sameWords += word
                        sameCount += 1
                    else:
                        if sameCount >= 3:
                            self.write("<b><font color=\"green\">" + sameWords + "</font></b>")
                        else:
                            self.write(sameWords)
                        sameCount = 0
                        sameWords = ""
                        self.write(word)
                if sameCount > 0:
                    if sameCount >= 2:
                        self.write("<b><font color=\"green\">" + sameWords + "</font></b>")
                    else:
                        self.write(sameWords)

                self.write('</td>')
                self.write('</tr>')

            i += 1

        self.write('</table>')
        self.write("</div>")
        self.write("</body></html>")

settings = {
    'static_path': os.path.join(os.path.dirname(__file__), 'anyou'),
    'cookie_secret': '61oETzKXQAGaYdkL5gEmGeJJFuYh7EQnp2XdTP1o/Vo=',
    'login_url': '/login',
    'xsrf_cookies': False,
}

application = tornado.web.Application([
    (r'/', MainHandler),
    (r'/similarCaseDemo', SimilarCaseDemoHandler),
    (r'/anyou/(xingshi_classify\.json)', tornado.web.StaticFileHandler, dict(path=settings['static_path'])),
    (r'/anyou/(minshi_classify\.json)', tornado.web.StaticFileHandler, dict(path=settings['static_path'])),
], **settings)


def init_tf_model():
    global tf_sess
    global tf_gragh
    global text_data
    global skip_thought_model

    tf_gragh = tf.Graph()
    with tf_gragh.as_default():
        text_data = TextData(
            FLAGS.train_data_path, max_vocab_size=FLAGS.max_vocab_size, max_len=FLAGS.target_max_len)
        # if text_data.max_len > FLAGS.target_max_len:
        #     FLAGS.target_max_len = text_data.max_len
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


def infer_tf_model(infer_data, infer_matrix):
    global tf_sess
    global text_data
    global skip_thought_model
    global seg_a_word

    if not infer_data:
        return

    seg_a_word = SegCNAWord()

    with open(FLAGS.pred_src_path, 'w') as f:
        tmp_res = seg_a_word.seg_cont(infer_data)
        for tmp_i in tmp_res:
            f.write((tmp_i + ' ').encode('utf-8'))

    with open(infer_matrix, 'w') as f:
        pred_data = text_data.pro_tuple_data(FLAGS.pred_src_path, batch_size=FLAGS.pred_batch_size)
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


def main():
    global simi_log
    global server_port
    global thread_num
    global word_vec_dim
    # global top_num_simi
    global work_data_dir
    global work_src_file
    global work_src_matrix
    global work_test_matrix
    cp = ConfigParser.SafeConfigParser()
    cp.read('conf_aysimi_skipthought.conf')
    server_port = cp.get('server', 'port')

    thread_num = int(cp.get('simi_calc', 'thread_num'))
    # word_vec_dim = int(cp.get('simi_calc', 'word_vec_dim'))
    word_vec_dim = FLAGS.num_units
    # top_num_simi = int(cp.get('simi_calc', 'top_num_simi'))
    work_data_dir = cp.get('simi_calc', 'work_data_dir')
    work_src_file = cp.get('simi_calc', 'work_src_file')
    work_src_matrix = cp.get('simi_calc', 'work_src_matrix')
    work_test_matrix = cp.get('simi_calc', 'work_test_matrix')

    simi_log = FinalLogger('aysimi_skipthought.log')

    init_tf_model()

    simi_log.info('---start anyou simi skipthought server---')

    application.listen(server_port)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    main()
