# -*- coding:utf-8 -*-

import seg_cn_a_word

seg_list = seg_cn_a_word.seg_cont("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
print " ".join(seg_list)