from __future__ import absolute_import, print_function

import tvm
import topi
import sys
import numpy as np

# Config
display_on = False
in_l = 224
in_w = 224

if len(sys.argv) > 1:
    data_file = sys.argv[1]
else:
    data_file = None

# === Start computation
L = tvm.var('L') # Image length
W = tvm.var('W') # Image width

# Input image
_img = tvm.placeholder((L, W), name='img', dtype='int32')

# Input image with padding
img_padx = tvm.compute((L, W + 2), lambda i, j:
        tvm.select((j < W), _img[i, j], tvm.const(0, dtype=_img.dtype)),
        name='img_padx')

img = tvm.compute((L + 2, W + 2), lambda i, j:
        tvm.select((i < W), img_padx[i, j], tvm.const(0, dtype=_img.dtype)),
        name='img')

blur_x = tvm.compute((L, W), lambda i, j:
        (img[i, j] + img[i, j + 1] + img[i, j + 2]) / 3.0,
        name='blur_x')

blur_img = tvm.compute((L, W), lambda i, j:
        (blur_x[i, j] + blur_x[i + 1, j] + blur_x[i + 2, j]) / 3.0,
        name='blur_img')

blur_img_int = topi.cast(blur_img, dtype='int32')

# === End computation

# Scheduling
s = tvm.create_schedule(blur_img_int.op)

# Compilation
#print(tvm.lower(s, [_img, img_padx, img, blur_x, blur_img, blur_img_int],
#    simple_mode=True))
func = tvm.build(s, [_img, blur_img_int])
assert func

#print("------func code------")
#print(func.imported_modules[0].get_source())

def print_img(img):
    for i in range(in_w):
        for j in range(in_l):
            print("{0} ".format(img[i, j]), end="")
        print("")
    return

# Prepare data
if data_file:
    # Read image from file (TBA)
    print("WARNING: Not implemented reading image from file, " +
            "generating random image")

# Generate image
np.random.seed(0)
np_img = np.random.randint(0, 256, size=(in_w, in_l))

if display_on:
    print_img(np_img)
    print("===")

in_img = tvm.nd.array(np_img.astype(_img.dtype), tvm.cpu(0))
out_img = tvm.nd.array(np.zeros((in_w, in_l), dtype=_img.dtype), tvm.cpu(0))

# Execution
func(in_img, out_img)

if display_on:
    print_img(out_img.asnumpy())

print("Finished")
