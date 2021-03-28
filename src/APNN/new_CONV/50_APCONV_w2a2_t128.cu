/*
  1-bit BMMA code.
  Runs at 500TOPS for matrix size of 4096x4096x8192.
  Borrows largely from CUDA-SDK.

  By Boyuan
*/

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#include <helper_cuda.h>
#include <helper_functions.h>

#define CHUNK_K 4
#define SKEW 1
#define WARPS_PER_BLOCK 8
#define WARP_SIZE 32
#define THREADS_PER_BLOCK WARP_SIZE * WARPS_PER_BLOCK
#define CHUNK_LINE_BYTES CHUNK_K * sizeof(int4)
#define WARP_COPY_BYTES WARP_SIZE * sizeof(int4)
#define CHUNK_COPY_LINES_PER_WARP WARP_COPY_BYTES / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES WARP_SIZE / CHUNK_COPY_LINES_PER_WARP
#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4
#define WARP_ROW_TILES 8
#define WARP_COL_TILES 4
#define BLOCK_ROW_TILES WARP_ROW_TILES * BLOCK_ROW_WARPS
#define BLOCK_COL_TILES WARP_COL_TILES * BLOCK_COL_WARPS
#define M 8
#define N 8
#define K 128

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

using namespace nvcuda;
using namespace nvcuda::wmma::experimental;

typedef union {
  int4 vec;
  int a[4];
} U4;

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
  int X_bit_offset = (Height+2) * (Width+2) * CIN/128;
  int W_bix_offset = 9*CIN*COUT/128;
  int BIT = 2;
  int X_ROW_BIT = (Width+2)*CIN/128;
  int W_ROW_BIT = 9*(CIN/128);

  // if (blockIdx.x == 0 && threadIdx.x == 0) {
  //   // for(int i = 0; i<Height*Width*CIN/32*BIT; i++) {
  //   //   printf("X[%d]: %x\n", i, *((int*)X+i));
  //   // }  
  //   for(int i = 0; i<COUT*9*CIN/32; i++) {
  //     printf("W[%d]: %x\n", i, *((int*)W+i));
  //   }  
  // }

  extern __shared__ int4 shmem[][CHUNK_K+SKEW]; // TODO: Padding opportunity may exist here.
  wmma::fragment<wmma::accumulator, 8, 8, 128, int> c[WARP_COL_TILES]
    [WARP_ROW_TILES];
  wmma::fragment<wmma::matrix_a, M, N, K, precision::b1, wmma::row_major> a[WARP_COL_TILES];
  wmma::fragment<wmma::matrix_b, M, N, K, precision::b1, wmma::col_major> b[WARP_ROW_TILES];


  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_i = (block_pos/(COUT/64)) / (Width/8) * 8;
    const unsigned int block_j = (block_pos/(COUT/64)) % (Width/8) * 8;
    const unsigned int block_z = block_pos % (COUT/64) * 64;

    if (block_i >= Height) {
      break;
    }

    int image_starting_idx = block_i * Width * CIN/128 + block_j * CIN/128;

    for(int i=0; i < WARP_COL_TILES; i++)
      for(int j=0; j < WARP_ROW_TILES; j++)
        wmma::fill_fragment(c[i][j], 0);
    
    // On the K dimension, there are 9*CIN/128 element to solve.
    // This for loop computes [0,1,2,...,int(9*CIN/128/CHUNK_K)*CHUNK_K-1]. Next for loop computes [int(9*CIN/128/CHUNK_K)*CHUNK_K, ..., 9*CIN/128-1]
    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k+CHUNK_K < 9*CIN/128; tile_k += CHUNK_K) {

      int SHMEM_i = threadIdx.x/4;
      int row = SHMEM_i / 8;
      int col = SHMEM_i % 8;
      int t = threadIdx.x % 4;

      int sub_row = (tile_k+t)/(3*CIN/128);
      int sub_col = (tile_k+t)%(3*CIN/128);


      int GL_idx = image_starting_idx + row*X_ROW_BIT + col*CIN/128 + sub_row*X_ROW_BIT + sub_col;

      // if (block_pos == 0 && tile_k ==0 && SHMEM_i == 1) {
      //   printf("tile_k: %d, block_i: %d, block_j: %d, row: %d, col: %d, sub_row: %d, sub_col: %d, GL_idx: %d\n", tile_k, block_i, block_j, row, col, sub_row, sub_col, GL_idx);
      //   printf("X[17]: %x %x %x %x\n", *((int*)X+ 4*17), *((int*)X+ 4*17+1), *((int*)X+ 4*17+2), *((int*)X+ 4*17+3));
      // }
  

      shmem[SHMEM_i][t] = X[GL_idx];

      SHMEM_i += 64;
      shmem[SHMEM_i][t] = X[GL_idx+X_bit_offset];
      
      SHMEM_i += 64;
      int weight_load_idx = (block_z + threadIdx.x/4) * W_ROW_BIT + tile_k + t;
      shmem[SHMEM_i][t] = W[weight_load_idx];

      SHMEM_i += 64;
      shmem[SHMEM_i][t] = W[weight_load_idx+W_bix_offset];

      __syncthreads();

      // if (block_pos == 0 && warpId == 0 && laneId == 0) {
      //   for(int i = 128; i < 256; i++) {
      //     for(int j = 0; j < 16; j++) {
      //       int *tile_ptr = (int*)&shmem[0][0] + i*20 + j;
      //       printf("tile_k: %d, i: %d, j: %d, val: %x\n", tile_k, i, j, *tile_ptr);
      //     }
      //   }
      // }
  


      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / 2) * M * 4 + (i * M);
          const int4 *tile_ptr = &shmem[shmem_idx_a][k_step];

          wmma::load_matrix_sync(a[i], tile_ptr, (CHUNK_K + SKEW)*128);
          
        // if (block_pos == 0 && warpId == 0) {
        //   // printf("tile_k: %d, k_step: %d, shmem_idx_a: %d\n", tile_k, k_step, shmem_idx_a);
        //   for(int t = 0; t<a[i].num_elements; t++) {
        //       printf("tile_k: %d, k_step: %d, a[%d].x[%d]: %x\n", tile_k, k_step, i, t, a[i].x[t]);
        //   }
        // }

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = 128 +
                                   (WARP_ROW_TILES * N) * (warpId % 2) +
                                   (j * N);
              const int4 *tile_ptr = &shmem[shmem_idx_b][k_step * (K/128)];
              wmma::load_matrix_sync(b[j], tile_ptr, (CHUNK_K + SKEW)*128);
              // wmma::fill_fragment(b[j], 0x5);
            }
            // printf("ckpt4\n");

            // if (block_pos == 0 && warpId == 0 && laneId == 0 && tile_k == 0) {
            //   for(int t = 0; t<b[j].num_elements; t++) {
            //       printf("b[%d].x[%d]: %x\n", j, t, b[j].x[t]);
            //   }
            // }
            wmma::bmma_sync(c[i][j], a[i], b[j], c[i][j], bmmaBitOpAND);
          }
        }
      }
      __syncthreads();
      // if (block_pos == 0 && warpId == 0) {
      //   for(int t = 0; t<c[0][0].num_elements; t++) {
      //       printf("tile_k: %d, c[0][0].x[%d]: %d\n", tile_k, t, c[0][0].x[t]);
      //   }
      // }

    }

    // int *shmem_warp_tile_ptr = (int*)&shmem[0][0] +
    //                           (warpId / 2) * 128 * 32 +
    //                           (warpId % 2) * 64; // Will be used only when writing back D. May be moved outside the for loop. TODO.


//     // Store the D fragments to shared memory.
// #pragma unroll
//     for (int i = 0; i < WARP_COL_TILES; i++) {
// #pragma unroll
//       for (int j = 0; j < WARP_ROW_TILES; j++) {
//         int *tile_ptr = shmem_warp_tile_ptr + i*128*8 + j*8;
//         wmma::store_matrix_sync(tile_ptr, c[i][j], 128,  wmma::mem_row_major);
//       }
//     }

//     __syncthreads();

//     if (block_pos == 0 && warpId == 0 && laneId == 0) {
//       for(int i = 1; i < 2; i++) {
//         for(int j = 0; j < 2; j++) {
//           int *tile_ptr = (int*)&shmem[0][0] + i*128 + j;
//           printf("i: %d, j: %d, val: %d\n", i, j, *tile_ptr);
//         }
//       }
//     }



#pragma unroll
    for (int tile_k = int(9*CIN/128/CHUNK_K)*CHUNK_K; tile_k < 9*CIN/128; tile_k++) {
      int SHMEM_i = threadIdx.x/4;
      int row = SHMEM_i / 8;
      int col = SHMEM_i % 8;
      int t = threadIdx.x % 4;

      int sub_row = (tile_k)/(3*CIN/128);
      int sub_col = (tile_k)%(3*CIN/128);

      int GL_idx = image_starting_idx + row*X_ROW_BIT + col*CIN/128 + sub_row*X_ROW_BIT + sub_col;
      *((int*)&shmem[SHMEM_i][0] + t) = *((int*)&X[GL_idx] + t);
      SHMEM_i += 64;
      *((int*)&shmem[SHMEM_i][0] + t) = *((int*)&X[GL_idx+X_bit_offset] + t);
      
      SHMEM_i += 64;
      int weight_load_idx = (block_z + threadIdx.x/4) * W_ROW_BIT + tile_k;
      *((int*)&shmem[SHMEM_i][0] + t) = *((int*)&W[weight_load_idx] + t);
      
      SHMEM_i += 64;
      *((int*)&shmem[SHMEM_i][0] + t) = *((int*)&W[weight_load_idx+W_bix_offset] + t);
      
      __syncthreads();

      // if (block_pos == 0 && warpId == 0 && laneId == 0) {
      //   for(int i = 128; i < 131; i++) {
      //     for(int j = 0; j < 4; j++) {
      //       int *tile_ptr = (int*)&shmem[0][0] + i*20 + j;
      //       printf("tile_k: %d, i: %d, j: %d, val: %x\n", tile_k, i, j, *tile_ptr);
      //     }
      //   }
      // }

      // Compute a grid of C matrix tiles in each warp.

#pragma unroll
      for (int i = 0; i < WARP_COL_TILES; i++) {
        size_t shmem_idx_a = (warpId / 2) * M * 4 + (i * M);
        const int4 *tile_ptr = &shmem[shmem_idx_a][0];

        wmma::load_matrix_sync(a[i], tile_ptr, (CHUNK_K + SKEW)*128);

#pragma unroll
        for (int j = 0; j < WARP_ROW_TILES; j++) {
          if (i == 0) {
            // Load the B matrix fragment once, because it is going to be
            // reused against the other A matrix fragments.
            size_t shmem_idx_b = 128 +
                                  (WARP_ROW_TILES * N) * (warpId % 2) +
                                  (j * N);
            const int4 *tile_ptr = &shmem[shmem_idx_b][0];

            wmma::load_matrix_sync(b[j], tile_ptr, (CHUNK_K + SKEW)*128);
          }
          // printf("ckpt4\n");

          wmma::bmma_sync(c[i][j], a[i], b[j], c[i][j], bmmaBitOpAND);
        }
      }
      __syncthreads();
    }


#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES/2; j++) {
        for(int t = 0; t<c[i][j].num_elements; t++) {
            c[i][j].x[t] = c[i][j].x[t] + (c[i][j+WARP_ROW_TILES/2].x[t]<<16);
        }
      }
    }


    // if (block_pos == 0){ // && warpId == 4 && laneId == 0) {
    //   for(int t = 0; t<c[0][0].num_elements; t++) {
    //       printf("c[0][0].x[%d]: %d\n", t, c[0][0].x[t]);
    //   }
    // }
    // This pointer is used to access the C and D matrix tiles this warp computes.
    int *shmem_warp_tile_ptr = (int*)&shmem[0][0] +
                              (warpId / 2) * 128 * 32 +
                              (warpId % 2) * 64; // Will be used only when writing back D. May be moved outside the for loop. TODO.

    // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES/2; j++) {
        int *tile_ptr = shmem_warp_tile_ptr + i*128*8 + j*8;
        wmma::store_matrix_sync(tile_ptr, c[i][j], 128,  wmma::mem_row_major);
      }
    }

    __syncthreads();

    // if (block_pos == 0 && warpId == 0 && laneId == 0) {
    //   for(int i = 1; i < 2; i++) {
    //     for(int j = 0; j < 64; j++) {
    //       int *tile_ptr = (int*)&shmem[0][0] + i*128 + j;
    //       printf("i: %d, j: %d, val: %d\n", i, j, *tile_ptr);
    //     }
    //   }
    // }


    U4 tmp0;
    U4 tmp1;
    U4 tmp2;
    U4 tmp3;
    U4 val[4];

    int *shmem_warp_stream_ptr = (int*)&shmem[0][0]+threadIdx.x/16*4*128 + (threadIdx.x%16)*4;
#pragma unroll
    for(int i = 0; i < 4; i++) {
      tmp0.vec = *((int4*)shmem_warp_stream_ptr);
      tmp1.vec = *((int4*)shmem_warp_stream_ptr+16);
      tmp2.vec = *((int4*)shmem_warp_stream_ptr+64*32);
      tmp3.vec = *((int4*)shmem_warp_stream_ptr+64*32+16);
      val[i].a[0] = tmp0.a[0] + 2*tmp1.a[0] + 2*tmp2.a[0] + 4*tmp3.a[0];
      val[i].a[1] = tmp0.a[1] + 2*tmp1.a[1] + 2*tmp2.a[1] + 4*tmp3.a[1];
      val[i].a[2] = tmp0.a[2] + 2*tmp1.a[2] + 2*tmp2.a[2] + 4*tmp3.a[2];
      val[i].a[3] = tmp0.a[3] + 2*tmp1.a[3] + 2*tmp2.a[3] + 4*tmp3.a[3];
      shmem_warp_stream_ptr += 128;
      // if (block_pos == 0 && warpId == 0 && laneId == 0) {
      //   printf("tmp0: %d %d %d %d\n", tmp0.a[0], tmp0.a[1], tmp0.a[2], tmp0.a[3]);
      //   printf("tmp1: %d %d %d %d\n", tmp1.a[0], tmp1.a[1], tmp1.a[2], tmp1.a[3]);
      //   printf("tmp2: %d %d %d %d\n", tmp2.a[0], tmp2.a[1], tmp2.a[2], tmp2.a[3]);
      //   printf("tmp3: %d %d %d %d\n", tmp3.a[0], tmp3.a[1], tmp3.a[2], tmp3.a[3]);
      //   printf("val[i]: %d %d %d %d \n", val[i].a[0], val[i].a[1], val[i].a[2], val[i].a[3]);
      // }
    }



    int SHMEM_row = threadIdx.x/16*4;
    int SHMEM_col = threadIdx.x%16;
    int Output_row = SHMEM_row/8;
    int Output_col = SHMEM_row%8;

    int* dst_gmem_warp_stream_ptr = Output + block_i * Width * COUT + block_j*COUT + block_z 
              + Output_row*Width*COUT + Output_col*COUT
              + SHMEM_col*4;
    // if (block_pos == 0) {
    //   printf("block_i: %d, block_j: %d, block_z: %d, threadIdx.x: %d, Output_row: %d, Output_col: %d, idx: %d\n", block_i, block_j, block_z, 
    //     threadIdx.x, Output_row, Output_col, 
    //     block_i * Width * COUT + block_j*COUT + block_z 
    //     + Output_row*Width*COUT + Output_col*COUT
    //     + SHMEM_col*4);
    // }
#pragma unroll
    for(int i=0; i<4; i++) {
      *(int4*)dst_gmem_warp_stream_ptr = val[i].vec;
      SHMEM_row += 1;
      Output_row = SHMEM_row/8;
      Output_col = SHMEM_row%8;
      dst_gmem_warp_stream_ptr = Output + block_i * Width * COUT + block_j*COUT + block_z 
              + Output_row*Width*COUT + Output_col*COUT
              + SHMEM_col*4;
    }
    __syncthreads();
  }
}