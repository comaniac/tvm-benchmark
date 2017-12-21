from __future__ import absolute_import, print_function

import tvm
import numpy as np

# === Start computation
N = tvm.var('N') # Data set size
D = tvm.var('D') # Feature number
L = tvm.var('L') # Label number

label = tvm.placeholder((N, L), name='label')
data = tvm.placeholder((N, D), name='data')
weight = tvm.placeholder((L, D + 1), name='weight')

data_expand = tvm.compute((N, D + 1), lambda n, d:
        tvm.select((d < D), data[n, d], tvm.const(1, dtype=data.dtype)),
        name='data_expand')

rd = tvm.reduce_axis((0, D + 1), name='rd')
dot = tvm.compute((N, L), lambda n, l:
        tvm.sum(weight[l, rd] * data_expand[n, rd], axis=rd),
        name='dot')

factor = tvm.compute((N, L), lambda n, l: 1 / (1 + tvm.exp(-dot[n, l])),
        name='factor')

def argmax_combine(x, y):
    lhs = tvm.select((x[1] > y[1]), x[0], y[0])
    rhs = tvm.select((x[1] > y[1]), x[1], y[1])
    return lhs, rhs

def argmax_identity(t0, t1):
    return tvm.const(-1, t0), tvm.min_value(t1)

argmax = tvm.comm_reducer(argmax_combine, argmax_identity, name='argmax')
dummy_idx = tvm.compute((L, ), lambda l: l, name='dummy_idx')
rl = tvm.reduce_axis((0, L), name='rl')
pred_idx,mdis = tvm.compute((N, ), lambda n:
        argmax((dummy_idx[rl], factor[n, rl]), axis=rl),
        name='pred_idx')

rn = tvm.reduce_axis((0, N), name='rn')
err = tvm.compute((1, ), lambda i:
        tvm.sum(1, rn, label[rn, pred_idx[rn]] < 0.5),
        name='err')

# === End computation

# Scheduling
s = tvm.create_schedule([pred_idx.op, err.op])

# Compilation
#print(tvm.lower(s, [data, label, weight, dot, scale, gradient, new_weight],
#    simple_mode=True))
calc = tvm.build(s, [data, label, weight, err])
assert calc
