from __future__ import absolute_import, print_function

import tvm
import numpy as np

# === Start computation
N = tvm.var('N') # Data set size
D = tvm.var('D') # Feature number

label = tvm.placeholder((N, ), name='label')
data = tvm.placeholder((N, D), name='data')
weight = tvm.placeholder((D + 1, ), name='weight')

data_expand = tvm.compute((N, D + 1), lambda n, d:
        tvm.select((d < D), data[n, d], tvm.const(1, dtype=data.dtype)),
        name='data_expand')

rd = tvm.reduce_axis((0, D + 1), name='rd')
dot = tvm.compute((N, ), lambda n:
        tvm.sum(weight[rd] * data_expand[n, rd], axis=rd), name='dot')

pred = tvm.compute((N, ), lambda n:
        tvm.select((dot[n] > 0),
            tvm.const(1, dtype=label.dtype), tvm.const(-1, dtype=label.dtype)),
        name='pred')

rn = tvm.reduce_axis((0, N), name='rn')
err = tvm.compute((1, ), lambda _: tvm.sum(1, rn, label[rn] != pred[rn]),
        name='err')

# === End computation

# Scheduling
s = tvm.create_schedule([pred.op, err.op])

# Compilation
calc = tvm.build(s, [data, label, weight, err])
assert calc
