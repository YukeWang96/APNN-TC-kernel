/*
  1-bit BMMA code.
  Runs at 500TOPS for matrix size of 4096x4096x8192.
  Borrows largely from CUDA-SDK.

  By Boyuan
*/
#ifndef NEW_KERNEL_CONV
#define NEW_KERNEL_CONV

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#include <helper_cuda.h>
#include <helper_functions.h>

#include "param.h"

#define CHUNK_K 4
#define SKEW 0 
#define WARPS_PER_BLOCK 8
#define WARP_SIZE 32
#define THREADS_PER_BLOCK WARP_SIZE*WARPS_PER_BLOCK
#define CHUNK_LINE_BYTES CHUNK_K*sizeof(int4)
#define WARP_COPY_BYTES WARP_SIZE*sizeof(int4)
#define CHUNK_COPY_LINES_PER_WARP WARP_COPY_BYTES / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES WARP_SIZE / CHUNK_COPY_LINES_PER_WARP
#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4
#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2
#define BLOCK_ROW_TILES WARP_ROW_TILES*BLOCK_ROW_WARPS
#define BLOCK_COL_TILES WARP_COL_TILES*BLOCK_COL_WARPS
#define M 8
#define N 8
#define K 128



using namespace nvcuda;
using namespace nvcuda::wmma::experimental;

// Assume that Kernel size is 3x3.
// Assume CIN is 128.
__inline__  __device__
void Conv128Layer_new(Conv128LayerParam* p) {

  const int4 *W = (int4*) (p->filter_gpu);
  const int4 *X = (int4*) (p->input_gpu);
  int *Output = (int*) p->output_gpu;
  const int Height = p->input_height;
  const int Width =  p->input_width;
  const int CIN = p->input_channels;
  const int COUT =  p->output_channels;

  // GEMM Configuration
  int X_bit_offset = Height * Width * CIN/128;

  extern __shared__ int4 shmem[][CHUNK_K+SKEW]; // TODO: Padding opportunity may exist here.

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_i = (block_pos/(COUT/64)) / (Width/8) * 4;
    const unsigned int block_j = (block_pos/(COUT/64)) % (Width/8) * 8;
    const unsigned int block_z = block_pos % (COUT/64) * 64;

    if (block_i >= Height) {
      break;
    }

    int image_starting_idx = block_i * Width * CIN/128 + block_j * CIN/128;

    wmma::fragment<wmma::accumulator, 8, 8, 128, int> c[WARP_COL_TILES]
                                                     [WARP_ROW_TILES];

    for(int i=0; i < WARP_COL_TILES; i++)
      for(int j = 0; j < WARP_ROW_TILES; j++)
        wmma::fill_fragment(c[i][j], 0);
    
    if (threadIdx.x < 120) {
      int threadPart = threadIdx.x/60;
      int threadOffset = threadIdx.x%60;
      int GL_idx = threadPart * X_bit_offset + (threadOffset/10)*Width + threadOffset%10 + image_starting_idx;
      *(&shmem[128][0]+threadIdx.x) = X[GL_idx];
    }
    __syncthreads();

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < int(9*CIN/128/4); tile_k += CHUNK_K) {

      int SHMEM_i = threadIdx.x/4;
      int SHMEM_part = SHMEM_i / 32;
      int SHMEM_offset = SHMEM_i % 32;
      int feature_expand_idx = SHMEM_part * 15 * CIN/2 + (SHMEM_offset/8)*10*CIN/128 + (SHMEM_offset%8)*CIN/128;

      int t = threadIdx.x % 4;
      int thread_expand_idx = feature_expand_idx + (tile_k*4+t)/(3*CIN/128)*10*(CIN/128) + (tile_k*4+t)%(3*CIN/128);
      shmem[SHMEM_i][t] = *(&shmem[128][0]+thread_expand_idx);

      SHMEM_i += 64;
      int weight_load_idx = SHMEM_part * 9 * CIN * COUT / 128 + (block_z + SHMEM_offset) * 9 * CIN/128;
      int thread_load_idx = weight_load_idx + (tile_k*4 + t) * CIN/128;
      shmem[SHMEM_i][t] = W[thread_load_idx];

      __syncthreads();

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, precision::b1, wmma::row_major> a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, precision::b1, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / 2) * M * 4 + (i * M);
          const int4 *tile_ptr = &shmem[shmem_idx_a][k_step];

          wmma::load_matrix_sync(a[i], tile_ptr, (CHUNK_K + SKEW)*128);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = 64 +
                                   (WARP_ROW_TILES * N) * (warpId % 2) +
                                   (j * N);
              const int4 *tile_ptr = &shmem[shmem_idx_b][k_step * (K/128)];

              wmma::load_matrix_sync(b[j], tile_ptr, (CHUNK_K + SKEW)*128);
            }
            // printf("ckpt4\n");

            wmma::bmma_sync(c[i][j], a[i], b[j], c[i][j]);
          }
        }
      }
      __syncthreads();
    }

    // Needs special handle for the remaining K.

    // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        int *tile_ptr = (int*)&shmem[0][0] + warpId*8*64 + (i*4+j) * 64;
        wmma::store_matrix_sync(tile_ptr, c[i][j], 8,  wmma::mem_row_major);
      }
    }

    __syncthreads();

    if (threadIdx.x < 32) {
      int num1 = 0;
      int num2 = 0;
      for (int j = 0; j < 32; j++) {
        int tile_i = threadIdx.x%16/8;
        int element_i = (threadIdx.x%16)%8;  
        int tile_j = j%32/8;
        int element_j = (j%32)%8;
        int final_i = warpId * 8 + tile_i*4+tile_j;
        int final_j = element_i *8 + element_j;
        int v0 = *((int*)&shmem[0][0]+final_i*64+final_j);
        int v1 = *((int*)&shmem[0][0]+final_i*64+final_j+32);
        int v2 = *((int*)&shmem[0][0]+(final_i+32)*64+final_j);
        int v3 = *((int*)&shmem[0][0]+(final_i+32)*64+final_j+32);
        int tmp = v0 + 2*v1 + 2*v2 + 4*v3;
        int tmp1 = tmp&1;
        int tmp2 = tmp&2;
        num1 = (num1 << 1) | tmp1;
        num2 = (num2 << 1) | tmp2;
      }
      *(Output+(threadIdx.x/8)*Width + threadIdx.x%8) = num1;
      *(Output+(threadIdx.x/8)*Width + threadIdx.x%8+ Height*Width*COUT/32) = num2;
    }

    __syncthreads();
  }
}
#endif