/*
  Simplest BMMA code with SASS instructions.

  Written by Boyuan Feng.
*/

#include <cuda.h>
#include <iostream>


void BMM_Wrapper(int M, int N, int K) {
  unsigned *d_A = 0;
  unsigned *d_B = 0;
  int *d_C = 0;
  int sizeA = M*K/32;
  int sizeB = N*K/32;
  int sizeC = M*N;

  /* Allocate device memory for the matrices */
  if (cudaMalloc(reinterpret_cast<void **>(&d_A), sizeA * sizeof(d_A[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_B), sizeB * sizeof(d_B[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_C), sizeC * sizeof(d_C[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
  }

  CUmodule module;
  CUfunction kernel;

  cuModuleLoad(&module, "cubin/simplestBMMA_SASS.cubin");
  cuModuleGetFunction(&kernel, module, "kern");

  void * args[3] = {&d_A, &d_B, &d_C};
  cuLaunchKernel(kernel, 1, 1, 1, 
                 32, 1, 1, 
                 0, 0, args, 0);
  cudaDeviceSynchronize();


}

int main(){
  BMM_Wrapper(1024, 1024, 128*16);
  return 0;
}