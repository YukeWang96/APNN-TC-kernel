#ifndef INPUT128LAYER
#define INPUT128LAYER

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

// compress the input from 32-bit to 1-bit
// store in 1-bit with packed 32-bit unsigned int format.
__device__ __inline__ void In128Layer_new(In128LayerParam* p)
{
    GET_LANEID;
    GET_WARPID;

    const int act_bit = p->bitwidth;
    // how many blocks in total. p->input_height/8 * p->input_width/128
    const int gdx = STEP8(p->input_height);                     // x size: vertical.
    const int gdy = STEP128(p->input_width);                    // y size: horizontal.
    const int offset = (p->input_height)*(p->input_width);      // layerwise offset of INPUT before bit compression.
    const int offset_opt = PAD8(p->input_height)*STEP128(p->input_width)*128/32;        // layerwise offset of OUTPUT after bit compression.

    // 32 warps per block
    const int lx = (warpid >> 2);  // x index, vertical
    const int ly = (warpid & 0x3); // y index, horizontal

    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {

        const int bx = bid / gdy; // x index of the current block
        const int by = bid % gdy; // y index of the current block

        // iterate through all bits
        #pragma unroll
        for (int bitIdx = 0; bitIdx < act_bit; bitIdx++){
            // boundry check whether inside, otherwise set to -1
            uin32 f0 = ( (by*128+ly*32+laneid<(p->input_width)) && (bx*8+lx<(p->input_height)) )?
                        p->output_uin32_gpu[bitIdx*offset + (bx*8+lx)*(p->input_width)+by*128+ly*32+laneid]: 0;
                        // p->input_gpu[(bx*8+lx)*(p->input_width)+by*128+ly*32+laneid]:-1.0f;

            // printf("f0: %u\n", f0);
            // compressed, any thing outside boundry would be set to 0.
            // note that * f0 > 0 * in the 0/1 case. but >= 0 in 1/-1 case
            unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>0));
            // printf("r0: %u\n", r0);

            // output the results
            if (laneid==0){
                // compatability with new kernel interleaved data store.
                // x-axis vertical offset (bx*8+lx)*gdy*4 --> act_bit*( (bx*8+lx)*gdy*4 ) + bitIdx * gdy*4.
                // y-axis horzontial index by*4+ly --> by*4+ly.
                int loc = act_bit*( (bx*8+lx)*gdy*4 ) + bitIdx*gdy*4 + by*4+ly;
                p->output_gpu[loc] = r0;
                // p->output_gpu[bitIdx*offset_opt + (bx*8+lx)*gdy*4 + by*4 + ly] = r0;
            }
        }

    }
}

#endif