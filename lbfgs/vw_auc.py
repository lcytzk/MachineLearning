#!/usr/bin/python
import sys
import math


positive_vec = [0 for n in range(1000001)]
negative_vec = [0 for n in range(1000001)]

for line in sys.stdin:
    segs = line.strip('\n').split(',')
    if len(segs) < 2:
        continue

    score = 1 / (1+math.exp(-float(segs[1])))
    flag = int(segs[0])

    label = flag
    if label > 0:
        label = 1
    val = score
    if val < 0.0 or val > 1.0:
        continue
    new_val = int(val * 1000000)
    if label > 0:
        positive_vec[new_val] += 1
    else:
        negative_vec[new_val] += 1

sum_pos = 0
sum_neg = 0
area = 0
for i in range(1000000, -1, -1):
    area += (sum_pos * 2 + positive_vec[i]) * negative_vec[i] / 2
    sum_pos += positive_vec[i]
    sum_neg += negative_vec[i]

auc = float(area) / (sum_pos * sum_neg)

print str(auc)
