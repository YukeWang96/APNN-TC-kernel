#!/usr/bin/env python3
import os
import sys

M_N_K_list = [
    # 16,
    # 32,
    # 64,   
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    # 16384,
    # 32768,
    # 65536,
]

MLP_origin = [
    [1024,  8, 784],
    [1024,  8, 1024],
    [1024,  8, 1024],
    [10,    8, 1024]
]
origin = MLP_origin

# for m, n, k in MLP_origin:
#     os.system("./MLP {} {} {}".format(m, n, k))

for m in M_N_K_list:
    os.system("./MLP {} {} {}".format(m, m, m))