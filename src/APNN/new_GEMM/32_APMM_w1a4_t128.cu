#ifndef APMM_32
#define APMM_32

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#include <helper_cuda.h>
#include <helper_functions.h>

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 8
#define N 8
#define K 128

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#define CHUNK_K 1

#define CHUNK_LINE_BYTES (CHUNK_K * sizeof(int4))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 8
#define WARP_COL_TILES 4

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B
// matrix in shared memory to minimize possible bank conflicts. Before
// performing the nvcuda::wmma::mma_sync operation, the warp must load the
// matrix data using the nvcuda::wmma::load_matrix_sync operation. Although the
// memory access pattern is not specified for that function, each lane in the
// warp can read one or multiple matrix elements from different matrix rows or
// columns. For shared memory, such access can result in bank conflicts if
// different rows / columns of the matrix map to the same bank. By shifting each
// row and column by a few bytes, we make sure that they map to different banks,
// thus reducing the number of possible bank conflicts. The number of 32
// one-byte "uint8_t" elements is chosen as the minimum possible shift because
// we must keep each row and column 256-bit aligned, as required by
// nvcuda::wmma::load_matrix_sync.
#define SKEW 2 // Updated for int4

using namespace nvcuda;
using namespace nvcuda::wmma::experimental;

__device__ __inline__ 
void Fc128Layer_new(Fc128LayerParam* p) {

  const int4* A = (int4*) p->input_gpu;
  const int4* B = (int4*) p->weight_gpu;
  int* D = (int*) p->output_gpu;  

  const int A_height = p->input_height;
  const int A_width  = p->input_width;
  const int B_width = p->weight_width;

  const int M_GLOBAL = A_height*(p->act_bit);
  const int N_GLOBAL = B_width*(p->w_bit); // N_global >= 128 cause the illegal memeory access.
  const int K_GLOBAL = A_width;

  const int M_TILES = STEP8(M_GLOBAL);
  const int N_TILES = STEP8(N_GLOBAL);
  const int K_TILES = STEP128(K_GLOBAL);

  extern __shared__ int4 shmem_1[][CHUNK_K+SKEW]; // TODO: Padding opportunity may exist here.

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i = block_pos / (N_TILES/16) * 16;
    const unsigned int block_tile_j = block_pos % (N_TILES/16) * 16;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }

    wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_COL_TILES]
                                                     [WARP_ROW_TILES];

    for(int i=0; i < WARP_COL_TILES; i++)
      for(int j = 0; j < WARP_ROW_TILES; j++)
        wmma::fill_fragment(c[i][j], 0);
    
    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const int4 *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * (K_GLOBAL/128)] +
                                              M * (K_GLOBAL/128) * (warpId % 4) * 4)
                                           : (&B[block_tile_j * N * (K_GLOBAL/128)] +
                                              N * (K_GLOBAL/128) * (warpId % 4) * 4);

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Offset in shared memory from which the B matrix is stored.
      const size_t shmem_idx_b_off = BLOCK_COL_TILES * M; // TODO: This BLOCK_COL_TILES may be selected to improve performance. Maybe moved outside the for loop.

      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy
      // the B matrix.
      size_t shmem_idx =
          warpId < (WARPS_PER_BLOCK / 2)
              ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 4)
              : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 4 + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * (K/128) +
                                (laneId / CHUNK_COPY_LINE_LANES) * (K_GLOBAL/128)) +
                       (laneId % CHUNK_COPY_LINE_LANES); // (K/128), since K=128 in bit. int4 is 128 bit.
                       
      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
      for (int i = 0; i < (32 / CHUNK_COPY_LINES_PER_WARP); i++) {
        // Copy 16 bytes at once in each lane.
        *((int4 *)&shmem_1[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
            *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        lane_ptr = (int4 *)(lane_ptr +
                            (K_GLOBAL/128) * CHUNK_COPY_LINES_PER_WARP);
        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
      }

      __syncthreads();

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, precision::b1, wmma::row_major> a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, precision::b1, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / 2) * M * 4 + (i * M);
          const int4 *tile_ptr = &shmem_1[shmem_idx_a][k_step];

          wmma::load_matrix_sync(a[i], tile_ptr, (CHUNK_K + SKEW)*128);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off +
                                   (WARP_ROW_TILES * N) * (warpId % 2) +
                                   (j * N);
              const int4 *tile_ptr = &shmem_1[shmem_idx_b][k_step * (K/128)];

              wmma::load_matrix_sync(b[j], tile_ptr, (CHUNK_K + SKEW)*128);
            }
            // printf("ckpt4\n");

            wmma::bmma_sync(c[i][j], a[i], b[j], c[i][j], bmmaBitOpAND);
          }
        }
      }
      __syncthreads();
    }
    
    // This pointer is used to access the C and D matrix tiles this warp computes.
    int *shmem_warp_tile_ptr = (int*)&shmem_1[0][0] +
                              (warpId / 2) * SHMEM_STRIDE * M * 4 +
                              (warpId % 2) * SHMEM_OFFSET; // Will be used only when writing back D. May be moved outside the for loop. TODO.

    // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * M + j * N;
        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // if (warpId == 0 && laneId == 0 && blockIdx.x==0) {
    //   for(int i = 0; i < 2; i++) {
    //     for(int j = 0; j < 2; j++) {
    //       printf("%d ", *((int*)&shmem_1[0][0]+i*128+j));
    //     }
    //     printf("\n");
    //   }
    // }

    // This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
    // int *shmem_warp_stream_ptr = (int*)&shmem_1[0][0] + warpId * SHMEM_STRIDE * M; // Will be used only when writing back D. Maybe moved outside the for loop. TODO.
    size_t idx = warpId * 16 * 128 + laneId * 4;

    int *shmem_warp_stream_ptr = (int*)&shmem_1[0][0]+idx;

    int val[16];

    typedef union {
      int4 vec;
      int a[4];
    } U4;
    U4 tmp0;

#pragma unroll
    for (int i = 0; i < 16; i++) {
      tmp0.vec = *((int4*)shmem_warp_stream_ptr);

      // if (warpId == 0 && laneId == 0 && blockIdx.x==0) {
      //   for(int i = 0; i < 4; i++) {
      //     printf("tmp0.a[%d], %d ", 3-i, tmp0.a[3-i]);
      //   }
      //   printf("\n");
      //   for(int i = 0; i < 4; i++) {
      //     printf("tmp1.a[%d], %d ", 3-i, tmp1.a[3-i]);
      //   }
      //   printf("\n");
      // }
  
      val[i] = tmp0.a[0] + 2*tmp0.a[1] + 4*tmp0.a[2] + 8*tmp0.a[3];
      // printf("val0: %d, val1: %d\n", val[2*i], val[2*i+1]);
      shmem_warp_stream_ptr += 128;
    }

    __syncthreads();

    idx = warpId * 16 * 32 + laneId;
    shmem_warp_stream_ptr = (int*)&shmem_1[0][0]+idx;
#pragma unroll
    for(int i = 0; i < 16; i++) {
      *shmem_warp_stream_ptr = val[i];
      shmem_warp_stream_ptr += 32;
    }
    __syncthreads();

    // if (warpId == 0 && laneId == 0 && blockIdx.x==0) {
    //   for(int i = 0; i < 2; i++) {
    //     for(int j = 0; j < 2; j++) {
    //       printf("%d ", *((int*)&shmem_1[0][0]+i*64+j));
    //     }
    //     printf("\n");
    //   }
    // }

    shmem_warp_stream_ptr = (int*)&shmem_1[0][0]+warpId * 16 * 32 + laneId*4;

    // This warp's pointer to the C matrix data to copy memory from to shared memory. 
    // TODO: May be moved outside the for loop.
    size_t gmem_idx = block_tile_i*M*N_GLOBAL/4 + block_tile_j*N/4 + warpId*16*N_GLOBAL/4 + (laneId%8)*4 + (laneId/8)*N_GLOBAL/4;
    
    // Now that shared memory contains all the D tiles, stream them to global memory.
    int *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
    for (int i = 0; i < 4; i++) {
      *((int4 *)(dst_gmem_warp_stream_ptr + 4*i*N_GLOBAL/4)) =
      *((int4 *)(shmem_warp_stream_ptr + i*4*32));
    }

    __syncthreads();
  }
}

__device__ __inline__ 
void Out128Layer_new(Out128LayerParam* p){
  
  const int4* A = (int4*) p->input_gpu;
  const int4* B = (int4*) p->weight_gpu;
  int* D = (int*) p->output_gpu;  

  const int A_height = p->input_height;
  const int A_width  = p->input_width;
  const int B_width = p->weight_width;

  const int M_GLOBAL = A_height*(p->act_bit);
  const int N_GLOBAL = B_width*(p->w_bit); // N_global >= 128 cause the illegal memeory access.
  const int K_GLOBAL = A_width;

  const int M_TILES = STEP8(M_GLOBAL);
  const int N_TILES = STEP8(N_GLOBAL);
  const int K_TILES = STEP128(K_GLOBAL);

  extern __shared__ int4 shmem_1[][CHUNK_K+SKEW]; // TODO: Padding opportunity may exist here.

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i = block_pos / (N_TILES/16) * 16;
    const unsigned int block_tile_j = block_pos % (N_TILES/16) * 16;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }

    wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_COL_TILES]
                                                     [WARP_ROW_TILES];

    for(int i=0; i < WARP_COL_TILES; i++)
      for(int j = 0; j < WARP_ROW_TILES; j++)
        wmma::fill_fragment(c[i][j], 0);
    
    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const int4 *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * (K_GLOBAL/128)] +
                                              M * (K_GLOBAL/128) * (warpId % 4) * 4)
                                           : (&B[block_tile_j * N * (K_GLOBAL/128)] +
                                              N * (K_GLOBAL/128) * (warpId % 4) * 4);

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Offset in shared memory from which the B matrix is stored.
      const size_t shmem_idx_b_off = BLOCK_COL_TILES * M; // TODO: This BLOCK_COL_TILES may be selected to improve performance. Maybe moved outside the for loop.

      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy
      // the B matrix.
      size_t shmem_idx =
          warpId < (WARPS_PER_BLOCK / 2)
              ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 4)
              : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 4 + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * (K/128) +
                                (laneId / CHUNK_COPY_LINE_LANES) * (K_GLOBAL/128)) +
                       (laneId % CHUNK_COPY_LINE_LANES); // (K/128), since K=128 in bit. int4 is 128 bit.
                       
      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
      for (int i = 0; i < (32 / CHUNK_COPY_LINES_PER_WARP); i++) {
        // Copy 16 bytes at once in each lane.
        *((int4 *)&shmem_1[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
            *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        lane_ptr = (int4 *)(lane_ptr +
                            (K_GLOBAL/128) * CHUNK_COPY_LINES_PER_WARP);
        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
      }

      __syncthreads();

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, precision::b1, wmma::row_major> a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, precision::b1, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / 2) * M * 4 + (i * M);
          const int4 *tile_ptr = &shmem_1[shmem_idx_a][k_step];

          wmma::load_matrix_sync(a[i], tile_ptr, (CHUNK_K + SKEW)*128);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off +
                                   (WARP_ROW_TILES * N) * (warpId % 2) +
                                   (j * N);
              const int4 *tile_ptr = &shmem_1[shmem_idx_b][k_step * (K/128)];

              wmma::load_matrix_sync(b[j], tile_ptr, (CHUNK_K + SKEW)*128);
            }
            // printf("ckpt4\n");

            wmma::bmma_sync(c[i][j], a[i], b[j], c[i][j], bmmaBitOpAND);
          }
        }
      }
      __syncthreads();
    }
    
    // This pointer is used to access the C and D matrix tiles this warp computes.
    int *shmem_warp_tile_ptr = (int*)&shmem_1[0][0] +
                              (warpId / 2) * SHMEM_STRIDE * M * 4 +
                              (warpId % 2) * SHMEM_OFFSET; // Will be used only when writing back D. May be moved outside the for loop. TODO.

    // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * M + j * N;
        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // if (warpId == 0 && laneId == 0 && blockIdx.x==0) {
    //   for(int i = 0; i < 2; i++) {
    //     for(int j = 0; j < 2; j++) {
    //       printf("%d ", *((int*)&shmem_1[0][0]+i*128+j));
    //     }
    //     printf("\n");
    //   }
    // }

    // This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
    // int *shmem_warp_stream_ptr = (int*)&shmem_1[0][0] + warpId * SHMEM_STRIDE * M; // Will be used only when writing back D. Maybe moved outside the for loop. TODO.
    size_t idx = warpId * 16 * 128 + laneId * 4;

    int *shmem_warp_stream_ptr = (int*)&shmem_1[0][0]+idx;

    int val[16];

    typedef union {
      int4 vec;
      int a[4];
    } U4;
    U4 tmp0;

#pragma unroll
    for (int i = 0; i < 16; i++) {
      tmp0.vec = *((int4*)shmem_warp_stream_ptr);

      // if (warpId == 0 && laneId == 0 && blockIdx.x==0) {
      //   for(int i = 0; i < 4; i++) {
      //     printf("tmp0.a[%d], %d ", 3-i, tmp0.a[3-i]);
      //   }
      //   printf("\n");
      //   for(int i = 0; i < 4; i++) {
      //     printf("tmp1.a[%d], %d ", 3-i, tmp1.a[3-i]);
      //   }
      //   printf("\n");
      // }
  
      val[i] = tmp0.a[0] + 2*tmp0.a[1] + 4*tmp0.a[2] + 8*tmp0.a[3];
      // printf("val0: %d, val1: %d\n", val[2*i], val[2*i+1]);
      shmem_warp_stream_ptr += 128;
    }

    __syncthreads();

    idx = warpId * 16 * 32 + laneId;
    shmem_warp_stream_ptr = (int*)&shmem_1[0][0]+idx;
#pragma unroll
    for(int i = 0; i < 16; i++) {
      *shmem_warp_stream_ptr = val[i];
      shmem_warp_stream_ptr += 32;
    }
    __syncthreads();

    // if (warpId == 0 && laneId == 0 && blockIdx.x==0) {
    //   for(int i = 0; i < 2; i++) {
    //     for(int j = 0; j < 2; j++) {
    //       printf("%d ", *((int*)&shmem_1[0][0]+i*64+j));
    //     }
    //     printf("\n");
    //   }
    // }

    shmem_warp_stream_ptr = (int*)&shmem_1[0][0]+warpId * 16 * 32 + laneId*4;

    // This warp's pointer to the C matrix data to copy memory from to shared memory. 
    // TODO: May be moved outside the for loop.
    size_t gmem_idx = block_tile_i*M*N_GLOBAL/4 + block_tile_j*N/4 + warpId*16*N_GLOBAL/4 + (laneId%8)*4 + (laneId/8)*N_GLOBAL/4;
    
    // Now that shared memory contains all the D tiles, stream them to global memory.
    int *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
    for (int i = 0; i < 4; i++) {
      *((int4 *)(dst_gmem_warp_stream_ptr + 4*i*N_GLOBAL/4)) =
      *((int4 *)(shmem_warp_stream_ptr + i*4*32));
    }

    __syncthreads();
  }
}
#endif