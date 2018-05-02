#!/usr/bin/env python
# -*- coding: utf-8 -*-

from prodata.data_utils import TextData


def main():
    # fname = '../../data/9107.txt.a.word'
    # train_data = TextData(fname)
    # out_data = train_data.pro_triples_data(200)
    # for i, batch in enumerate(out_data):
    #     print i, batch

    import numpy as np
    a = [[1, 2, 3]]
    np.array(a).partition()
    pass


if __name__ == '__main__':
    main()
    print 'ok'