from __future__ import absolute_import, print_function

import tvm
import sys
import numpy as np

# === Start computation
N = tvm.var('N') # Data set size
D = tvm.var('D') # Feature number

x = tvm.placeholder((N, D), name='x')
y = tvm.placeholder((N, ), name='y')
w = tvm.placeholder((D + 1, ), name='w')

x_expand = tvm.compute((N, D + 1), lambda n, d:
        tvm.select((d < D), x[n, d], tvm.const(1, dtype=x.dtype)),
        name='x_expand')

rd = tvm.reduce_axis((0, D + 1), name='rd')
predict = tvm.compute((N, ), lambda n:
        tvm.sum(w[rd] * x_expand[n, rd], axis=rd),
        name='predict')

rn = tvm.reduce_axis((0, N), name='rn')
err = tvm.compute((1,), lambda i:
        tvm.sum(((predict[rn] - y[rn]) * (predict[rn] - y[rn])) / N, axis=rn),
        name='err')

# === End computation

# Scheduling
s = tvm.create_schedule(err.op)

# Compilation
calc = tvm.build(s, [x, y, w, err])
assert calc
