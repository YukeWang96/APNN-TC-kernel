/*
  Basic functionality of BMMA in CUDA Programming.

  Written by Boyuan Feng.
*/

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

using namespace nvcuda;
using namespace nvcuda::wmma::experimental;

__global__ void BMM(unsigned *A, unsigned *B, int *C, int A_height, int A_width, int B_width)
{ // This function borrows largely from "Listing 3: BMM baseline implementation"
  //     from "Accelerating Binarized Neural Networks via Bit-Tensor-Cores in Turing GPUs", TPDS'20.
  int bx = blockIdx.x*blockDim.y+threadIdx.y; int by = blockIdx.y;
  wmma::fragment<wmma::matrix_a, 8,8,128, precision::b1, wmma::row_major> a_frag; //tile A
  wmma::fragment<wmma::matrix_b,8,8,128,precision::b1, wmma:col_major> b_frag; //tile B
  wmma::fragment<wmma::accumulator,8,8,128,int> c_frag;
  wmma::fill_fragment(c_frag,0);
  for (int i = 0; i < (A_width/128); i++) {
    load_matrix_sync(a_frag, A+(bx*8*A_width + i*128)/32, A_width); // fetch tile A
    load_matrix_sync(a_frag, B+(by*8*A_width + i*128)/32, A_width); // fetch tile B
    bmma_sync(c_frag, a_frag, b_frag, c_frag);  // BMM
  }
  for (int i=0; i<c_frag.num_elements;i++) c_frag.x[i]=A_width-2*c_frag.x[i];
  store_matrix_sync(C+(bx*8*B_width+by*8), c_frag, B_width, wmma::mem_row_major); // store tile C
}

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
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_B), sizeB * sizeof(d_B[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_C), sizeC * sizeof(d_C[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
    return EXIT_FAILURE;
  }

  // START: Added for measuring speed.
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // END: Added for measuring speed.
  for(int trial = 0; trial < 200; trial ++) {
    /* Performs operation using cublas */
    BMM<<<dim3(M/16, N/8), dim3(32,2)>>>(d_A, d_B, d_C, M, K, N);
  }

  // START: MEASURE time and flops.

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;

  cudaEventElapsedTime(&milliseconds, start, stop);

  // printf("Time: %f ms\n", milliseconds);
  printf("M: %d, N: %d, K: %d, TFLOPS: %.2f\n", 8192, 8192, 8192, static_cast<double>(200*(static_cast<double>(8192) *
                                                8192 * 8192 * 2) /
                                               (milliseconds / 1000.)) /
                               1e12);

  // END: MEASURE time and flops.

  if (cudaFree(d_A) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(d_B) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(d_C) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }
}

int main(){
  BMM_Wrapper(4096, 4096, 4096);
  return 0;
}