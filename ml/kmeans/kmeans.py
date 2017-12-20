from __future__ import absolute_import, print_function

import tvm
import numpy as np
import mse

# Config
in_n = 819200
in_c = 5
in_v = 34
iter_time = 10

# === Start computation
N = tvm.var('N') # Data set size
V = tvm.var('V') # Feature number
C = tvm.var('C') # Center number

data = tvm.placeholder((N, V), name='data')
center = tvm.placeholder((C, V), name='center')

# Compute distances
rv = tvm.reduce_axis((0, V), name='rv')
dis = tvm.compute((N, C), lambda n, c: tvm.sum(
    (data[n, rv]-center[c, rv])*(data[n, rv]-center[c, rv]), axis=rv),
    name='dis')

# Determine the center
def argmin_combine(x, y):
    lhs = tvm.select((x[1] <= y[1]), x[0], y[0])
    rhs = tvm.select((x[1] <= y[1]), x[1], y[1])
    return lhs, rhs

def argmin_identity(t0, t1):
    return tvm.const(-1, t0), tvm.max_value(t1)

argmin = tvm.comm_reducer(argmin_combine, argmin_identity, name='argmin')
rc = tvm.reduce_axis((0, C), name='rc')
dummy_idx = tvm.compute((C, ), lambda c: c, name='dummy_idx')
idx,mdis = tvm.compute((N, ), lambda i:
        argmin((dummy_idx[rc], dis[i, rc]), axis=rc), name='idx_w_dis')
        
# Update the center
rn2 = tvm.reduce_axis((0, N), name='rn2')
center_cnt = tvm.compute((C,), lambda c: tvm.sum(1, rn2, idx[rn2] == c),
    name='center_cnt')

rn1 = tvm.reduce_axis((0, N), name='rn1')
new_center = tvm.compute((C, V), lambda c, v: tvm.sum(
    data[rn1, v] / center_cnt[c], rn1, idx[rn1] == c), name='new_center')

# === End computation

# Scheduling
s = tvm.create_schedule([idx.op, new_center.op])

# Compilation
#print(tvm.lower(s, [data, center, dis, idx,
#    center_cnt, new_center], simple_mode=True))
func = tvm.build(s, [data, center, new_center])
assert func

#print("------func code------")
#print(func.imported_modules[0].get_source())

# Generate data
in_data = tvm.nd.array(np.random.uniform(
    size=(in_n, in_v), low=0.0, high=1.0).astype(data.dtype), tvm.cpu(0))
in_center = tvm.nd.array(np.random.uniform(
    size=(in_c, in_v), low=0.0, high=1.0).astype(center.dtype), tvm.cpu(0))

# Evaluation
evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 1)
last_mse = float('inf')
has_error = False
for i in range(iter_time):
    out_new_center = tvm.nd.array(np.zeros((in_c, in_v), dtype=data.dtype),
            tvm.cpu(0))
    out_mse = tvm.nd.array(np.zeros(1, dtype='float64'), tvm.cpu(0))
   
    t = evaluator(in_data, in_center, out_new_center).mean
    mse.calc(in_data, out_new_center, out_mse)
    curr_mse = out_mse.asnumpy()[0]
    delta_mse = curr_mse - last_mse
    print("Iteration {0} ({1}s), delta MSE = {2}".format(i + 1, t, delta_mse))
    if delta_mse > 0:
        has_error = True
    last_mse = curr_mse
    in_center = out_new_center

if has_error:
    print("Error: MSE increased!")
print("Finished")
