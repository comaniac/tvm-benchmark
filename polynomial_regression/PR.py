from __future__ import absolute_import, print_function

import tvm
import numpy as np
import verify

# Config
display_on = False
iter_time = 100
in_n = 20000
in_d = 784
learning_rate = 50e-8

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
dot = tvm.compute((N, ), lambda n:
        tvm.sum(w[rd] * x_expand[n, rd], axis=rd),
        name='dot')

rn = tvm.reduce_axis((0, N), name='rn')
gradient = tvm.compute((D + 1, ), lambda d:
        tvm.sum((dot[rn] - y[rn]) * x_expand[rn, d] * (2.0 / N), axis=rn),
        name='gradient')

new_w = tvm.compute((D + 1, ), lambda d: w[d] - learning_rate * gradient[d],
        name='new_w')

# === End computation

# Scheduling
s = tvm.create_schedule(new_w.op)

# Compilation
#print(tvm.lower(s, [x, y, x_expand, w, dot, gradient, new_w],
#    simple_mode=True))
func = tvm.build(s, [x, y, w, new_w])
assert func

#print("------func code------")
#print(func.imported_modules[0].get_source())

# Generate x
np_x = np.random.uniform(size=(in_n, in_d), low=0, high=100)
golden_w = np.random.uniform(size=in_d, low=-1, high=1)
noise = np.random.uniform(size=in_n, low=-5, high=5)
np_y = np.array([golden_w.dot(np_x[i]) + noise[i] for i in range(in_n)])

in_y = tvm.nd.array(np_y.astype(y.dtype), tvm.cpu(0))
in_x = tvm.nd.array(np_x.astype(x.dtype), tvm.cpu(0))
in_w = tvm.nd.array(np.zeros(in_d + 1, dtype=w.dtype), tvm.cpu(0))

# Evaluation
display_animation = display_on
if display_on and in_d != 1:
    display_animation = False
    print("WARNING: Will only display the MSE trend " +
            "due to high dimensional data points.")

if display_on:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    fig = plt.figure()
    ax = plt.axes()
    if display_animation and len(np_x) > 512:
        print("WARNING: Only display centrids due to too many data points.")

mse_history = []

def run(i):
    global in_x
    global in_y
    global in_w
    global mse_history
    out_w = tvm.nd.array(np.zeros(in_d + 1, dtype=w.dtype),
            tvm.cpu(0))
    func(in_x, in_y, in_w, out_w)
    in_w = out_w
    out_err = tvm.nd.array(np.zeros(1, dtype='float32'), tvm.cpu(0))
    verify.calc(in_x, in_y, in_w, out_err)
    err = out_err.asnumpy()[0]
    mse_history.append(err)
    print("Round {0}, MSE {1}".format(i, err))
   
    if display_animation:
        np_w = out_w.asnumpy()
        ax.clear()
        if len(np_x) <= 512:
            ax.scatter(np_x, np_y, c='b', s=20)
        draw_x = np.arange(0, 100, 0.1)
        draw_y = np.array([np_w[0] *_x + np_w[1] for _x in draw_x])
        ax.plot(draw_x, draw_y, 'r-')
    return []

if display_animation:
    anim = animation.FuncAnimation(fig, run, frames=iter_time, interval=1000,
            blit=True, repeat=False)
    plt.show()
else:
    for i in range(iter_time):
        run(i)

if display_on:
    ax.clear()
    ax.plot(np.arange(0, iter_time), mse_history)
    plt.show()

print("Finished")
