from __future__ import absolute_import, print_function

import tvm
import numpy as np

N = tvm.var('N') # Data set size
V = tvm.var('V') # Feature number
C = tvm.var('C') # Center number

data = tvm.placeholder((N, V), name='data')
center = tvm.placeholder((C, V), name='center')

# === Start computation
# Compute distances
rv = tvm.reduce_axis((0, V), name='rv')
dis = tvm.compute((N, C), lambda n, c: tvm.sum(
    (data[n, rv]-center[c, rv])*(data[n, rv]-center[c, rv]), axis=rv),
    name='dis')

rc = tvm.reduce_axis((0, C), name='rc')
mse_n = tvm.compute((N,), lambda n: tvm.sum(dis[n, rc], axis=rc), name='mse_n')
rn = tvm.reduce_axis((0, N), name='rn')
mse = tvm.compute((1,), lambda i: tvm.sum(mse_n[rn], axis=rn), name='mse')

# === End computation

# Scheduling
s = tvm.create_schedule(mse.op)

# Compilation
calc = tvm.build(s, [data, center, mse], "llvm")
assert calc
