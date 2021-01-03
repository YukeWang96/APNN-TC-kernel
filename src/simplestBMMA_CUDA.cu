/*
  Simplest BMMA code.

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

__global__ void BMM(unsigned *A, unsigned *B, int *C)
{
  wmma::fragment<wmma::matrix_a, 8,8,128, precision::b1, wmma::row_major> a_frag; //tile A
  wmma::fragment<wmma::matrix_b, 8,8,128, precision::b1, wmma::col_major> b_frag; //tile B
  wmma::fragment<wmma::accumulator,8,8,128,int> c_frag;
  wmma::fill_fragment(c_frag,0);
  load_matrix_sync(a_frag, A, 128); // fetch tile A
  load_matrix_sync(b_frag, B, 128); // fetch tile B
  bmma_sync(c_frag, a_frag, b_frag, c_frag);  // BMM
  store_matrix_sync(C, c_frag, 8, wmma::mem_row_major); // store tile C
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
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_B), sizeB * sizeof(d_B[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_C), sizeC * sizeof(d_C[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
  }


  BMM<<<1, 32>>>(d_A, d_B, d_C);

}

int main(){
  BMM_Wrapper(8, 8, 128);
  return 0;
}