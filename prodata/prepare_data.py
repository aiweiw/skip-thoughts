#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import

import codecs
import os
import re
import time
import sys

from segcont.seg_cn_a_word import SegCNAWord

re_file_name = re.compile(r'^90(96|63)')
re_sub = re.compile(r'\s')

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')


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


def mix_list_data():
    # anyou_file_path = os.path.expanduser('~/fastText/anyou/minshi/src')
    anyou_file_path = os.path.expanduser('../data/src')
    anyou_train_data = '../data/mix.txt.6396'

    obj_src = None
    obj_tgt = None
    try:
        obj_tgt = codecs.open(anyou_train_data, 'w', 'utf-8')
        seq_file = list_file(anyou_file_path)
        for file_i in seq_file:
            if not re_file_name.match(file_i):
                continue
            print(file_i)
            file_src = anyou_file_path + '/' + file_i
            obj_src = codecs.open(file_src, 'r', 'utf-8')
            line = obj_src.readline()
            line_num = 1
            while line and line_num <= 1000:
                obj_tgt.write(re_sub.sub('', line))
                obj_tgt.write('\n')
                # print(file_i + ': ' + str(line_num))
                line = obj_src.readline()
                line_num += 1
            obj_src.close()
            time.sleep(0)

        obj_tgt.close()
    except Exception, e:
        print Exception, e
    finally:
        if obj_src and not obj_src.closed:
            obj_src.close()
        if obj_tgt and not obj_tgt.closed:
            obj_tgt.close()


def seg_data():
    # anyou_file_data = os.path.expanduser('~/fastText/anyou/minshi/src/9015.txt')
    anyou_file_data = os.path.expanduser('../data/src/9015.txt')
    anyou_train_data = '../data/9015.txt.a.word'

    obj_src = None
    obj_tgt = None

    seg_a_word = SegCNAWord()
    try:
        obj_tgt = codecs.open(anyou_train_data, 'w', 'utf-8')
        obj_src = codecs.open(anyou_file_data, 'r', 'utf-8')

        line = obj_src.readline()
        src_line_num = 0
        while line and src_line_num < 10000:
            tmp_res = seg_a_word.seg_cont(re_sub.sub('', line))
            for tmp_i in tmp_res:
                obj_tgt.write((tmp_i + ' ').encode('utf-8'))
            obj_tgt.write('\n')
            print('---going--- ', src_line_num)
            line = obj_src.readline()
            src_line_num += 1

        obj_src.close()
        obj_tgt.close()
    except Exception, e:
        print Exception, e
    finally:
        if obj_src and not obj_src.closed:
            obj_src.close()
        if obj_tgt and not obj_tgt.closed:
            obj_tgt.close()


if __name__ == '__main__':
    # mix_list_data()
    seg_data()
    print('ok')
