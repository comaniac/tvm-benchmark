from __future__ import absolute_import, print_function

import tvm
import sys
import numpy as np
import verify

# Config
display_on = False
iter_time = 20
in_n = 20000
in_l = 10
in_d = 784
learning_rate = 50e-8

if len(sys.argv) > 1:
    data_file = sys.argv[1]
else:
    data_file = None

# === Start computation
N = tvm.var('N') # Data set size
D = tvm.var('D') # Feature number
L = tvm.var('L') # Label number

label = tvm.placeholder((N, L), name='label')
data = tvm.placeholder((N, D), name='data')
weight = tvm.placeholder((L, D), name='weight')

rd = tvm.reduce_axis((0, D), name='rd')
dot = tvm.compute((N, L), lambda n, l:
        tvm.sum(weight[l, rd] * data[n, rd], axis=rd),
        name='dot')

scale = tvm.compute((N, L), lambda n, l:
        (1 / (1 + tvm.exp(-label[n, l] * dot[n, l])) - 1) * label[n, l],
        name='scale')

rn = tvm.reduce_axis((0, N), name='rn')
gradient = tvm.compute((L, D), lambda l, d:
        tvm.sum(scale[rn, l] * data[rn, d], axis=rn),
        name='gradient')

new_weight = tvm.compute((L, D), lambda l, d:
        weight[l, d] - learning_rate * gradient[l, d],
        name='new_weight')

# === End computation

# Scheduling
s = tvm.create_schedule(new_weight.op)

# Compilation
#print(tvm.lower(s, [data, label, weight, dot, scale, gradient, new_weight],
#    simple_mode=True))
func = tvm.build(s, [data, label, weight, new_weight])
assert func

#print("------func code------")
#print(func.imported_modules[0].get_source())

# Prepare data
if data_file:
    # Read from file
    np_data = []
    np_label = []
    with open(data_file, 'r') as f:
        for i in range(in_n):
            line = f.readline()
            raw = line.split(' ')
            np_label.append([int(l) for l in raw[:in_l]])
            np_data.append([float(d) for d in raw[in_l:in_l + in_d]])
    
    np_label = np.array(np_label)
    np_data = np.array(np_data)
else:
    # Generate data
    np_data = np.random.uniform(size=(in_n, in_d), low=-1, high=3)
    golden_weight = np.random.uniform(size=(in_l, in_d), low=-1, high=1)
    noise = np.random.uniform(size=(in_n), low=-0.2, high=0.2)
    np_label = []
    for n in range(in_n):
        tmp = []
        for l in range(in_l):
            dot = golden_weight[l].dot(np_data[n])
            tmp.append(1 / (1 + np.exp(-dot)) + noise[n])
        idx = np.argmax(tmp)
        np_label.append([1 if i == idx else -1 for i in range(in_l)])
    np_label = np.array(np_label)

in_label = tvm.nd.array(np_label.astype(label.dtype), tvm.cpu(0))
in_data = tvm.nd.array(np_data.astype(data.dtype), tvm.cpu(0))
in_weight = tvm.nd.array(np.zeros((in_l, in_d), dtype=weight.dtype),
        tvm.cpu(0))

# Evaluation
err_rate_history = []
for i in range(iter_time):
    out_weight = tvm.nd.array(np.zeros((in_l, in_d), dtype=weight.dtype),
            tvm.cpu(0))
    func(in_data, in_label, in_weight, out_weight)
    in_weight = out_weight
    out_err = tvm.nd.array(np.zeros(1, dtype='int32'), tvm.cpu(0))
    verify.calc(in_data, in_label, in_weight, out_err)
    err_rate = float(out_err.asnumpy()[0]) / in_n
    print("Round {0}, error rate {1}%".format(i, 100.0 * err_rate))
    err_rate_history.append(err_rate)

if display_on:
    print("Generating learning trend graph...")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(np.arange(0, iter_time), err_rate_history)
    plt.show()

print("Finished")
