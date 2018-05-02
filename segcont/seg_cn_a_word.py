#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import, unicode_literals

import re

from ._compat import *

re_han = re.compile('([\u4E00-\u9FD5]+)', re.U)
re_skip = re.compile('(\t\r\n\f\v|\s)', re.U)
re_blk = re.compile('(^[ ]+$)', re.U)


class SegCNAWord(object):
    """
    Chinese A Word Segmentation
    """
    def __init__(self):
        pass

    def seg_cont(self, sentence):
        assert sentence is not None

        sentence = strdecode(sentence)
        blocks = re_han.split(sentence)

        for blk in blocks:
            if not blk or re_blk.match(blk):
                continue
            if re_han.match(blk):
                for word in self.__cut_every(blk):
                    yield word
            else:
                tmp = re_skip.split(blk)
                for x in tmp:
                    if not x or re_blk.match(x):
                        continue
                    yield x

    def __cut_every(self, sentence):
        dag = self.__get_DAG(sentence)
        old_j = -1
        for k, L in iteritems(dag):
            if len(L) == 1 and k > old_j:
                yield sentence[k:L[0] + 1]
                old_j = L[0]
            else:
                for j in L:
                    if j > k:
                        yield sentence[k:j + 1]
                        old_j = j

    def __get_DAG(self, sentence):
        DAG = {}
        N = len(sentence)
        for k in xrange(N):
            tmplist = []
            i = k
            flag = True
            while i < N and flag:
                tmplist.append(i)
                flag = False
                i += 1
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

dt = SegCNAWord()

seg_cont = dt.seg_cont
