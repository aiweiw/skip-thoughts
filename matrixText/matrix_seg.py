# coding=utf-8
import codecs

import jieba.posseg as pseg
import os
import time
import sys
import re
import urllib
import urllib2
import node

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

sys.path.append('../..')

"""
segment
input anyou-file
1. anyou root-node file
2. anyou every file
"""

patt_time = re.compile(r'^(((\s?\d\s?){4}年(\s?\d\s?){1,2}月(\s?\d\s?){1,2}日)|((\s?\d\s?){4}年(\s?\d\s?){1,2}月)|'
                       r'((\s?\d\s?){4}年)|((\s?\d\s?){1,2}月(\s?\d\s?){1,2}日)|((\s?\d\s?){1,2}日))'
                       r'(底|初|末|以来|年底|年初|年末|月底|月初|月末|开始|期间|份)?$')

patt_num = re.compile(r'\d+\s*\，\s*\d+')

patt_sym = re.compile(r'\，\s*\w+\s*[\。\，]')

patt_seg = re.compile(r'\。|\？|\！|\，')


def list_file(path):
    """
    :param path:
    :return:
    """
    file_list = []
    files = os.listdir(path)
    for f in files:
        if f[0] == '.':
            pass
        else:
            file_list.append(f)
    return file_list


def seg_cont_jieba(to_seg_content):
    """
    :param to_seg_content:
    :return:
    """
    # ansj-update-segment
    if to_seg_content is None:
        return None

    seg_words = pseg.cut(to_seg_content)
    return seg_words


def seg_cont_ansj(to_seg_content, ansj_serve_url=None, out_filter=0):
    """
    :param to_seg_content:
    :param ansj_serve_url:
    :param out_filter: 0 - seg_vec, 1 - seg_filter, * - seg_vec, seg_filter
    :return:
    """
    if not to_seg_content or not ansj_serve_url:
        return None

    seg_ontent = dict()
    seg_ontent['segTent'] = to_seg_content
    data_urlencode = urllib.urlencode(seg_ontent)
    req = urllib2.Request(ansj_serve_url, data=data_urlencode)
    response = urllib2.urlopen(req)
    time.sleep(0.001)
    seg_res = response.read()

    seg_vec_tfidf = seg_res.split('-SEGMENT-')

    if out_filter == 0:
        # return segseg-words full
        return seg_vec_tfidf[0]
    elif out_filter == 1:
        # return seg-words filter
        return seg_vec_tfidf[1]
    else:
        # return seg-words full+filter
        return seg_vec_tfidf[0], seg_vec_tfidf[1]

    # seg_vec = seg_vec_tfidf[0].split()
    # seg_filter = seg_vec_tfidf[1].split()

    # if out_filter == 0:
    #     # return segseg-words full
    #     return seg_vec
    # elif out_filter == 1:
    #     # return seg-words filter
    #     return seg_filter
    # else:
    #     # return seg-words full+filter
    #     return seg_vec, seg_filter


def seg_sent_base(sentence):
    """
    :param sentence:
    :return:
    """
    match = re.compile(r'\。|\.|\，|\,')
    res_segment = match.split(sentence)

    return res_segment


def seg_sentence(sentence, min_words_num=5):
    """ Segment sentence

    Args:
        sentence: sentence
        min_words_num: a sentence contains min_words_num words
    Returns:
        segment sentence
    """
    if not sentence:
        return None

    if not isinstance(sentence, str):
        sentence = str(sentence)

    num_seg_all = patt_num.findall(sentence)
    while num_seg_all:
        for num_one in num_seg_all:
            sentence = sentence.replace(num_one, num_one.replace('，', ','))
        num_seg_all = patt_num.findall(sentence)

    sym_seg_all = patt_sym.findall(sentence)
    while sym_seg_all:
        for sym_one in sym_seg_all:
            sentence = sentence.replace(sym_one, sym_one.replace('，', ',', 1))
        sym_seg_all = patt_sym.findall(sentence)

    res_segment = patt_seg.split(sentence)
    res_seg_len = len(res_segment)

    min_sent_len = min_words_num * len('。')
    i = 0
    while i < res_seg_len:
        if patt_time.match(res_segment[i].replace(' ', '')) or len(res_segment[i].replace(' ', '')) < min_sent_len:
            if len(res_segment) == 1:
                break
            if i < len(res_segment) - 1:
                res_segment[i] += ',' + res_segment[i + 1]
                res_segment.remove(res_segment[i + 1])
            else:
                res_segment[i - 1] += ',' + res_segment[i]
                res_segment.remove(res_segment[i])
            res_seg_len = len(res_segment)
        else:
            i += 1

    return res_segment


def seg_sentence_err(sentence, min_words_num=5):
    """ Segment sentence

    Args:
        sentence: sentence
        min_words_num: a sentence contains min_words_num words
    Returns:
        segment sentence
    """
    if not sentence:
        return None

    if not isinstance(sentence, str):
        sentence = str(sentence)

    num_seg_all = patt_num.findall(sentence)
    while num_seg_all:
        for num_one in num_seg_all:
            sentence = sentence.replace(num_one, num_one.replace('，', ','))
        num_seg_all = patt_num.findall(sentence)

    sym_seg_all = patt_sym.findall(sentence)
    while sym_seg_all:
        for sym_one in sym_seg_all:
            sentence = sentence.replace(sym_one, sym_one.replace('，', ',', 1))
        sym_seg_all = patt_sym.findall(sentence)

    res_segment = patt_seg.split(sentence)
    res_seg_len = len(res_segment)

    min_sent_len = min_words_num * len('。')
    for i in range(res_seg_len):
        if i >= res_seg_len:
            break
        if patt_time.match(res_segment[i].replace(' ', '')) or len(res_segment[i].replace(' ', '')) < min_sent_len:
            if len(res_segment) == 1:
                break
            if i < len(res_segment) - 1:
                res_segment[i] += ',' + res_segment[i + 1]
                res_segment.remove(res_segment[i + 1])
            else:
                res_segment[i - 1] += ',' + res_segment[i]
                res_segment.remove(res_segment[i])

            res_seg_len = len(res_segment)

    return res_segment


"""
generate-one-file-seg-words-for-word2vec & compute-tfidf
ansj or jieba-seg
"""


def seg_file(to_seg_path, seg_file_name, save_seg_path, jieba_ansj='jieba', ansj_serve_url=None):
    """
    :param to_seg_path:
    :param seg_file_name:
    :param save_seg_path:
    :param jieba_ansj:
    :param ansj_serve_url:
    :return:
    """
    if not seg_file_name or not os.path.exists(to_seg_path):
        return

    if sys.getdefaultencoding() != 'utf-8':
        reload(sys)
        sys.setdefaultencoding('utf8')

    if to_seg_path[len(to_seg_path) - 1] != '/':
        to_seg_path += '/'

    if not os.path.exists(save_seg_path):
        os.mkdir(save_seg_path)

    file_to_seg = None
    file_to_save = None
    try:
        file_to_seg = open(to_seg_path + seg_file_name, 'r')
        line = file_to_seg.readline()

        result = list()
        result_filter = list()
        while line:
            if jieba_ansj == 'ansj' and ansj_serve_url:
                # ansj-segment
                # segContent = dict()
                # segContent['segTent'] = line
                # data_urlencode = urllib.urlencode(segContent)
                # req = urllib2.Request(ansj_serve_url, data=data_urlencode)
                # response = urllib2.urlopen(req)
                # time.sleep(0.001)
                # seg_res = response.read()
                #
                # seg_vec_tfidf = seg_res.split('-SEGMENT-')
                # seg_vec = seg_vec_tfidf[0].split()
                # seg_tfidf = seg_vec_tfidf[1].split()

                seg_vec_jion, seg_tfidf_join = seg_cont_ansj(line, ansj_serve_url, 2)
                seg_vec = seg_vec_jion.strip(' \r\n').split()
                seg_tfidf = seg_tfidf_join.strip(' \r\n').split()

                for word_to_vec in seg_vec:
                    word = ''.join(word_to_vec.split())
                    if word != '' and word != '\n' and word != '\n\n':
                        result.append(word)

                for word_to_tfidf in seg_tfidf:
                    word = ''.join(word_to_tfidf.split())
                    if word != '' and word != '\n' and word != '\n\n':
                        result_filter.append(word)
            else:
                # jieba-segment
                seg_list = seg_cont_jieba(line)
                for word, flag in seg_list:
                    word = ''.join(word.split())
                    if word != '' and word != '\n' and word != '\n\n':
                        result.append(word)

            line = file_to_seg.readline()
        file_to_seg.close()
        time.sleep(0)

        date_span = re.search(r'\d+', seg_file_name).span()
        if save_seg_path[len(save_seg_path) - 1] != '/':
            save_seg_path += '/'

        if jieba_ansj == 'ansj':
            file_seg = save_seg_path + seg_file_name[date_span[0]:date_span[1]] + '.txt.ansj.seg'
            file_to_save = open(file_seg, 'w')
            file_to_save.write(' '.join(result))
            file_to_save.close()
            time.sleep(0)

            file_seg = save_seg_path + seg_file_name[date_span[0]:date_span[1]] + '.txt.ansj.seg.filter'
            file_to_save_filter = open(file_seg, 'w')
            file_to_save_filter.write(' '.join(result_filter))
            file_to_save_filter.close()
            time.sleep(0)
        else:
            file_seg = save_seg_path + seg_file_name[date_span[0]:date_span[1]] + '.txt.jieba.seg'
            file_to_save = open(file_seg, 'w')
            file_to_save.write(' '.join(result))
            file_to_save.close()
            time.sleep(0)
    except Exception, e:
        print Exception, e
    finally:
        if file_to_seg and not file_to_seg.closed:
            file_to_seg.close()
        if file_to_save and not file_to_save.closed:
            file_to_save.close()


"""
generate-anyou-seg-words-for-word2vec & compute-tfidf
ansj-seg
"""


def seg_xml_ansj(xml_file, to_seg_path, save_seg_file, ansj_serve_url=None):
    """
    :param xml_file:
    :param to_seg_path:
    :param save_seg_file:
    :param ansj_serve_url:
    :return:
    """
    if not os.path.exists(xml_file) or not os.path.exists(to_seg_path) or not ansj_serve_url:
        return

    if sys.getdefaultencoding() != 'utf-8':
        reload(sys)
        sys.setdefaultencoding('utf8')

    anyou_firstlist = []
    anyou_nodemap = {}
    anyou_label_map = {}

    node.loadConfig(xml_file, anyou_firstlist, anyou_nodemap, anyou_label_map)

    anyou_dict_txt = {}
    for anyou in anyou_firstlist:
        anyou_list_txt = []
        node.get_all_grandsons_id(anyou, anyou_list_txt)
        anyou_key = anyou.get('DM')
        anyou_dict_txt[anyou_key] = list(set(anyou_list_txt))

    for key, val in anyou_dict_txt.items():

        # if key != '9000':
        #     continue

        if not os.path.exists(save_seg_file):
            os.mkdir(save_seg_file)

        if save_seg_file[len(save_seg_file) - 1] != '/':
            save_seg_file += '/'
        train_data_file = save_seg_file + key + '.txt.ansj.learn'

        try:
            train_file = open(train_data_file, 'w')
            for v in val:
                tmp_file = to_seg_path + str(v) + '.txt'
                train_data_file_one = save_seg_file + str(v) + '.txt.ansj.seg'
                if os.path.exists(tmp_file):
                    train_file_one = open(train_data_file_one, 'w')
                    tmp_file_read = open(tmp_file)
                    line = tmp_file_read.readline()

                    while line:
                        #
                        # segContent = dict()
                        # segContent['segTent'] = line
                        # data_urlencode = urllib.urlencode(segContent)
                        # req = urllib2.Request(ansj_serve_url, data=data_urlencode)
                        # response = urllib2.urlopen(req)
                        # time.sleep(0.001)
                        # seg_res = response.read()
                        #
                        # seg_vec_tfidf = seg_res.split('-SEGMENT-')
                        # seg_vec = seg_vec_tfidf[0].split()
                        # seg_tfidf = seg_vec_tfidf[1].split()

                        seg_vec_jion, seg_tfidf_join = seg_cont_ansj(line, ansj_serve_url, 2)
                        seg_vec = seg_vec_jion.strip(' \r\n').split()
                        seg_tfidf = seg_tfidf_join.strip(' \r\n').split()

                        for word_to_vec in seg_vec:
                            word = ''.join(word_to_vec.split())

                            train_file.write((word + ' ').encode('utf-8'))
                            train_file.flush()

                        train_file.write(('\n').encode('utf-8'))
                        train_file.flush()

                        for word_to_tfidf in seg_tfidf:
                            word = ''.join(word_to_tfidf.split())

                            train_file_one.write((word + ' ').encode('utf-8'))
                            train_file_one.flush()

                        line = tmp_file_read.readline()

                    train_file_one.close()
                    time.sleep(0)
                    tmp_file_read.close()
                    time.sleep(0)
                    print 'on...'

            train_file.close()
            time.sleep(0)
            print '---going---'

        except Exception, e:
            print Exception, e
        finally:
            if train_file and not train_file.closed:
                train_file.close()
            if tmp_file_read and not tmp_file_read.closed:
                tmp_file_read.close()
            if train_file_one and not train_file_one.closed:
                train_file_one.close()



def seg_xml_ansj_dm(xml_file, to_seg_path, save_seg_file, ansj_serve_url=None):
    """
    :param xml_file:
    :param to_seg_path:
    :param save_seg_file:
    :param ansj_serve_url:
    :return:
    """
    if not os.path.exists(xml_file) or not os.path.exists(to_seg_path) or not ansj_serve_url:
        return

    if sys.getdefaultencoding() != 'utf-8':
        reload(sys)
        sys.setdefaultencoding('utf8')

    anyou_firstlist = []
    anyou_nodemap = {}
    anyou_label_map = {}

    node.loadConfig(xml_file, anyou_firstlist, anyou_nodemap, anyou_label_map)

    anyou_dict_txt = {}
    for key in anyou_nodemap.keys():
        if key is None:
            continue
        anyou_list_txt = []
        node.get_all_grandsons_id(anyou_nodemap.get(key), anyou_list_txt)
        anyou_dict_txt[key] = list(set(anyou_list_txt))

    for key, val in anyou_dict_txt.items():

        if not os.path.exists(save_seg_file):
            os.mkdir(save_seg_file)

        if save_seg_file[len(save_seg_file) - 1] != '/':
            save_seg_file += '/'
        train_data_file = save_seg_file + key + '.txt.ansj.learn'

        if_create_train_file = False
        for v in val:
            if_file = to_seg_path + str(v) + '.txt'
            if os.path.exists(if_file):
                if_create_train_file = True
                break
        if not if_create_train_file:
            continue

        try:
            # train_file = open(train_data_file, 'w')
            train_file = codecs.open(train_data_file, 'a', 'utf-8')
            for v in val:
                tmp_file = to_seg_path + str(v) + '.txt'
                if os.path.exists(tmp_file):
                    print '---------- (key, val): (', key, v, ')'
                    tmp_file_read = open(tmp_file)
                    line = tmp_file_read.readline()

                    while line:

                        seg_vec_jion = seg_cont_ansj(line, ansj_serve_url)
                        seg_vec = seg_vec_jion.strip(' \r\n').split()

                        for word_to_vec in seg_vec:
                            word = ''.join(word_to_vec.split())

                            train_file.write((word + ' ').encode('utf-8'))
                            train_file.flush()

                        train_file.write(('\r\n').encode('utf-8'))
                        train_file.flush()

                        line = tmp_file_read.readline()

                    tmp_file_read.close()
                    time.sleep(0)
                    print 'on...'

            train_file.close()
            time.sleep(0)
            print '---going---'

        except Exception, e:
            print Exception, e
        finally:
            if train_file and not train_file.closed:
                train_file.close()
            if tmp_file_read and not tmp_file_read.closed:
                tmp_file_read.close()


"""
generate-anyou-seg-words-for-word2vec & compute-tfidf
jieba-seg
"""


def seg_xml_jieba(xml_file, to_seg_path, save_seg_file):
    """
    :param xml_file:
    :param to_seg_path:
    :param save_seg_file:
    :return:
    """
    if not os.path.exists(xml_file) or not os.path.exists(to_seg_path):
        return

    if sys.getdefaultencoding() != 'utf-8':
        reload(sys)
        sys.setdefaultencoding('utf8')

    anyou_firstlist = []
    anyou_nodemap = {}
    anyou_label_map = {}

    node.loadConfig(xml_file, anyou_firstlist, anyou_nodemap, anyou_label_map)

    anyou_dict_txt = {}
    for anyou in anyou_firstlist:
        anyou_list_txt = []
        node.get_all_grandsons_id(anyou, anyou_list_txt)
        anyou_key = anyou.get('DM')
        anyou_dict_txt[anyou_key] = list(set(anyou_list_txt))

    for key, val in anyou_dict_txt.items():
        if not os.path.exists(save_seg_file):
            os.mkdir(save_seg_file)

        if to_seg_path[len(save_seg_file) - 1] != '/':
            save_seg_file += '/'
        train_data_file = save_seg_file + key + '.txt.learn'

        try:
            train_file = open(train_data_file, 'w')
            for v in val:
                tmp_file = to_seg_path + str(v) + '.txt'
                train_data_file_one = save_seg_file + str(v) + '.txt.seg'
                if os.path.exists(tmp_file):
                    train_file_one = open(train_data_file_one, 'w')
                    tmp_file_read = open(tmp_file)
                    line = tmp_file_read.readline()
                    while line:
                        words = seg_cont_jieba(line)

                        for word, flag in words:
                            if word == '\n':
                                train_file.write(word.encode('utf-8'))
                                train_file.flush()
                                # train_file_one.write(word.encode('utf-8'))
                                # train_file_one.flush()
                            else:
                                train_file.write((word + ' ').encode('utf-8'))
                                train_file.flush()
                                train_file_one.write((word + ' ').encode('utf-8'))
                                train_file_one.flush()
                        line = tmp_file_read.readline()
                    train_file_one.close()
                    time.sleep(0)
                    tmp_file_read.close()
                    time.sleep(0)
                    print 'on...'

            train_file.close()
            time.sleep(0)
            print '---going---'

        except Exception, e:
            print Exception, e
        finally:
            if train_file and not train_file.closed:
                train_file.close()
            if tmp_file_read and not tmp_file_read.closed:
                tmp_file_read.close()
            if train_file_one and not train_file_one.closed:
                train_file_one.close()
