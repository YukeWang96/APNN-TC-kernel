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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void BMM(unsigned *A, unsigned *B, int *C, int A_height, int A_width, int B_width)
{ // This function copys "Listing 3: BMM baseline implementation"
  //     from "Accelerating Binarized Neural Networks via Bit-Tensor-Cores in Turing GPUs", TPDS'20.
  int bx = blockIdx.x*blockDim.y+threadIdx.y; int by = blockIdx.y;
  wmma::fragment<wmma::matrix_a, 8,8,128, precision::b1, wmma::row_major> a_frag; //tile A
  wmma::fragment<wmma::matrix_b,8,8,128,  precision::b1, wmma::col_major> b_frag; //tile B
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

__global__ void BMM_SHMEM(unsigned *A, unsigned *B, int *C, int A_height, int A_width, int B_width)
{ // This function copys "Listing 4: Bit-Matrix-Multiplication"
  //    from "Accelerating Binarized Neural Networks via Bit-Tensor-Cores in Turing GPUs", TPDS'20.
  __shared__ uint4 As[32], Bs[32]; // buffering (8*128)*8 bit block in shared memory
  const int laneid=threadIdx.x; const int wx = threadIdx.y; const int wy=threadIdx.z; // tile index
  const int bx = blockIdx.x; const int by=blockIdx.y; // block index
  wmma::fragment<wmma::matrix_a,8,8,128, precision::b1, wmma::row_major> a_frag; // tile A
  wmma::fragment<wmma::matrix_b,8,8,128, precision::b1, wmma::col_major> b_frag; // tile B
  wmma::fragment<wmma::accumulator,8,8,128, int> c_frag; wmma::fill_fragment(c_frag,0); // tile C
  for (int k=0; k<A_width; k=k+128) { // Bug in Listing 4. Change from "k++" => "k+=128"
    if(wx==0 && wy==0) {
      // one warp fetches data into shared memory for 16 warps of a thread block
      As[laneid] = ((uint4*)A)[((bx*32+laneid)*A_width+k)/128]; // Bug in Listing 4. Additionally "/128".
      Bs[laneid] = ((uint4*)B)[((by*32+laneid)*A_width+k)/128]; // Bug in Listing 4. Additionally "/128".
    }
    __syncthreads(); // for respecting RAW dependency
    load_matrix_sync(a_frag, &As[wx*8], 128);
    load_matrix_sync(b_frag, &Bs[wy*8], 128);
    bmma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads(); // for respecting WAR dependency
  }
  for (int i=0; i<c_frag.num_elements;i++) c_frag.x[i] = (A_width*128)-(2*c_frag.x[i]); // +1/-1 BMM
  store_matrix_sync(&C[(bx*4+wx)*8*B_width + (by*4+wy)*8], c_frag, B_width, wmma::mem_row_major);
}


void BMM_Wrapper(int M, int N, int K) {
  unsigned *d_A = 0;
  unsigned *d_B = 0;
  int *d_C = 0;
  int sizeA = M*K/32;
  int sizeB = N*K/32;
  int sizeC = M*N;

  int NUM_PROFILE = 200;

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

  // START: Performance measurement for BMM
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for(int trial = 0; trial < NUM_PROFILE; trial ++) {
    BMM<<<dim3(M/16, N/8), dim3(32,2)>>>(d_A, d_B, d_C, M, K, N);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds_BMM = 0;

  cudaEventElapsedTime(&milliseconds_BMM, start, stop);

  printf("BMM Performance. M: %d, N: %d, K: %d, TOPS: %.2f\n", M, N, K, static_cast<double>(NUM_PROFILE*(static_cast<double>(M) *
                                                N * K * 2) /
                                               (milliseconds_BMM / 1000.)) /
                               1e12);

  // END: End Performance measurement for BMM

  // START: Performance measurement for BMM_SHMEM
  cudaEvent_t start_BMM_SHMEM, stop_BMM_SHMEM;
  cudaEventCreate(&start_BMM_SHMEM);
  cudaEventCreate(&stop_BMM_SHMEM);
  cudaEventRecord(start_BMM_SHMEM);

  for(int trial = 0; trial < NUM_PROFILE; trial ++) {
    BMM_SHMEM<<<dim3(M/32, N/32), dim3(32,4,4)>>>(d_A, d_B, d_C, M, K, N);
  }

  cudaEventRecord(stop_BMM_SHMEM);
  cudaEventSynchronize(stop_BMM_SHMEM);

  float milliseconds_BMM_SHMEM = 0;
  cudaEventElapsedTime(&milliseconds_BMM_SHMEM, start_BMM_SHMEM, stop_BMM_SHMEM);
  printf("BMM_SHMEM. M: %d, N: %d, K: %d, TOPS: %.2f\n", M, N, K, static_cast<double>(NUM_PROFILE*(static_cast<double>(M) *
                                                N * K * 2) /
                                               (milliseconds_BMM_SHMEM / 1000.)) /
                               1e12);
  // END: Performance measurement for BMM_SHMEM

  if (cudaFree(d_A) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (A)\n");
  }

  if (cudaFree(d_B) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (B)\n");
  }

  if (cudaFree(d_C) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (C)\n");
  }
}

int main(){
  BMM_Wrapper(8192, 8192, 8192);
  return 0;
}