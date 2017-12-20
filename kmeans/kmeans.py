from __future__ import absolute_import, print_function

import tvm
import numpy as np
import mse

# Config
display_on = False
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
converged = False

if display_on and in_v != 2:
    print("WARNING: Turn off display because it only allows 2-D data points.")
    display_on = False

if display_on:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
    np_data = in_data.asnumpy()
    if len(np_data) > 512:
        print("WARNING: Only display centrids due to too many data points.")

def run(i):
    global last_mse
    global converged
    global evaluator
    global in_data
    global in_center
    global np_data
    out_new_center = tvm.nd.array(np.zeros((in_c, in_v), dtype=data.dtype),
            tvm.cpu(0))
    out_mse = tvm.nd.array(np.zeros(1, dtype='float64'), tvm.cpu(0))
    t = evaluator(in_data, in_center, out_new_center).mean
    mse.calc(in_data, out_new_center, out_mse)
    curr_mse = out_mse.asnumpy()[0]
    delta_mse = curr_mse - last_mse
    print("Round {0} ({1}s), delta MSE = {2}".format(i + 1, t, delta_mse))
    if delta_mse > 0:
        converged = True
    last_mse = curr_mse
    in_center = out_new_center

    if display_on:
        np_center = out_new_center.asnumpy()
        ax.clear()
        if len(np_data) <= 512:
            ax.scatter(np_data[:, 0], np_data[:, 1], c='b')
        ax.scatter(np_center[:, 0], np_center[:, 1], c='r', s=100)
    return []

if display_on:
    anim = animation.FuncAnimation(fig, run, frames=iter_time, interval=1000,
            blit=True, repeat=False)
    plt.show()
else:
    for i in range(iter_time):
        run(i)
        if converged:
            break

print("Finished")
