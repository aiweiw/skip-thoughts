#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import codecs
import sys

from segcont.seg_cn_a_word import SegCNAWord


def main():

    if sys.getdefaultencoding() != 'utf-8':
        reload(sys)
        sys.setdefaultencoding('utf8')

    # src_file = '/home/uww/fastText/matTextData/data/segAnsj/1.txt.ansj.learn'
    # src_file = '/home/uww/Work/Projgram/PyProj/skip-thought-tf/data/9107.txt.ansj.learn'
    src_file = '../../data/aytrain.txt'
    tgt_file = '../../data/aytrain.txt.a.word'
    test_seg = SegCNAWord()
    src_obj = None
    tgt_obj = None
    try:
        src_obj = codecs.open(src_file, 'r', 'utf-8')
        tgt_obj = codecs.open(tgt_file, 'w', 'utf-8')

        src_line = src_obj.readline()
        num_line = 1
        while src_line:
            tmp_res = test_seg.seg_cont(src_line)
            for tmp_i in tmp_res:
                tgt_obj.write((tmp_i + ' ').encode('utf-8'))
            # tgt_obj.write('\n'.encode('utf-8'))
            src_line = src_obj.readline()
            print '---going---', num_line
            num_line += 1

        src_obj.close()
        tgt_obj.close()
    except Exception, e:
        print Exception, e
    finally:
        if src_obj and not src_obj.closed:
            src_obj.close()
        if tgt_obj and not tgt_obj.closed:
            tgt_obj.close()


if __name__ == '__main__':
    main()
    print 'ok'

