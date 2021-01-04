# Machine Information

Num of Registers / SM: 255. Same as Volta.

## Several difference between sm80 (A100) and sm86

Maximum number of thread blocks per SM: 32 for sm80 devices, 16 for sm 86 devices.
Shared memory per SM: 164KB for sm 80 devices, 100KB for sm 86 devices.
Maximum shared memory per thread block: 163 KB for sm 80 devices, 99 KB for sm 86 devices.

# Basic Units

## Frag_C
Frac_C stores 8*8 elements of type int32.
It requires 256 bytes in total.
To store Frac_C, all 32 threads in a warp provide 64 registers in total (2 register per thread, each register is 4 bytes), leading to 64 * 4 = 256 bytes.

## Frag_A and Frag_B
Frag_A stores 8*128 elements of type 1 bit.
It requires 8*128/8 = 128 bytes.
To store Frag_A, all 32 threads in a warp provide 32 registers in total (1 register per thread, each register is 4 bytes), leading to 32 * 4 = 128 bytes.


Frag_B is the same.








