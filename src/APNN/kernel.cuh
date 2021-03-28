// ---------------------------------------------------------------------------
// File: kernel.cuh
// TC-BNN bit-fully-connected and bit-convolution GPU kernel functions.
// ---------------------------------------------------------------------------
// See our arXiv paper for detail: https://arxiv.org/abs/2006.16578
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/TCBNN
// PNNL-IPID: 31925-E, ECCN: EAR99, IR: PNNL-SA-152850
// BSD Lincese.
// Richland, 99352, WA, USA. June-30-2020.
// ---------------------------------------------------------------------------

#ifndef KERNEL_CUH
#define KERNEL_CUH
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <assert.h>

#include "param.h"

// * quantization of a single float value
__device__ __inline__ uin32 quantize(float val, int bitwidth){
    const int max_val = 10000;
    const int min_val = -max_val;
    if (val > max_val) val = max_val - 1;
    if (val < min_val) val = min_val + 1;
    uin32 ans = (val - min_val) * (1 << bitwidth) / (max_val - min_val); 
    return ans;
}

// compress the input from 32-bit to 1-bit
// store in 1-bit with packed 32-bit unsigned int format.
__device__ __inline__ void In128Layer(In128LayerParam* p)
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

__device__ __inline__ void Fc128Layer(Fc128LayerParam* p)
{
    using namespace nvcuda;
    GET_LANEID;
    GET_WARPID;
    using namespace nvcuda::wmma::experimental;

    const int act_bit = p->act_bit;
    const int w_bit = p->w_bit;
    // printf("act_bit: %d, w_bit: %d\n", act_bit, w_bit);
    
    // layerwise offset measured in uin32
    const int act_offset = PAD8(p->input_height)*STEP128(p->input_width)*128/32;
    const int w_offset = STEP128(p->weight_height)*PAD128(p->weight_width)*128/32;
    const int opt_offset = PAD8(p->input_height)*STEP128(p->weight_width)*128/32;

    //__shared__ int Cs[32][64];
    // M x N x K
    extern __shared__ int Cs[];
    const int gdx = STEP8(p->input_height);     // vertical     --> M
    const int gdy = STEP8(p->weight_width);     // horizontal   --> N
    const int gdk = STEP128(p->input_width);    // iterations   --> K
    const int gdm = STEP128(p->weight_width);   // output width ---> N

    // each grid with gridim.x blocks, 
    // each block with 32 warps.
    // each warp processes each 8x8 tile
    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy; bid+=gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> tmp_frag;

        wmma::fill_fragment(c_frag, 0);

        // rwo major output.
        const int bx = bid / gdy;
        const int by = bid % gdy;

        // iterate along different bits.
        for (int bit = 0; bit < act_bit * w_bit; bit++){
            int b_act = bit % act_bit;
            int b_w = bit / act_bit;
            int b_opt = b_act + b_w;

            // accmuluation of the current bit.
            wmma::fill_fragment(tmp_frag, 0);

            // iterate along the K columns
            for (int i=0; i<gdk; i++)
            {
                // iterate along the K diemsnion by loading the tile from the 
                load_matrix_sync(a_frag, p->input_gpu + b_act*act_offset + bx*8*gdk*4 + i*128/32, gdk*128);
                load_matrix_sync(b_frag, p->weight_gpu + b_w*w_offset + by*8*gdk*4 + i*128/32, gdk*128);
                bmma_sync(tmp_frag, a_frag, b_frag, tmp_frag, bmmaBitOpAND);
            }

            // Accumulation.
            #pragma unroll
            for (int t = 0; t < tmp_frag.num_elements; t++) {
                c_frag.x[t] += tmp_frag.x[t] << b_opt;
            }
        }

        // quantization at the fragment into act_bit (stored in uint32).
        #pragma unroll
        for (int t = 0; t < c_frag.num_elements; t++) {
            // printf("%d\n", c_frag.x[t]);
            c_frag.x[t] = quantize(c_frag.x[t], act_bit);
        }

        // finished one output tile and store to shared memory
        store_matrix_sync(&Cs[warpid*64], c_frag, 8, wmma::mem_row_major);

        #pragma unroll
        for (int bIdx = 0; bIdx < act_bit; bIdx++){
            
            // change to 8-bit address
            uin8* Cb = (uin8*)(&(p->output_gpu[bIdx*opt_offset])); 

            // 2D index of a warp
            const int gy = (laneid%8);
            const int gx = (laneid/8);

            // checking position constraints.
            bool v0_in = ((by*8+gy)<(p->output_width)) && ((bx*8+gx)<(p->output_height));
            bool v1_in = ((by*8+gy)<(p->output_width)) && ((bx*8+gx+4)<(p->output_height)); 

            // get the corresponding decomposed bit value.
            bool v0 = v0_in && (((Cs[warpid*64+laneid]>>bIdx) & 0x1) > 0);
            bool v1 = v1_in && (((Cs[warpid*64+32+laneid]>>bIdx) & 0x1) > 0);

            // printf("v0: %d, v1: %d\n", v0, v1); //ok, all 1s
            // bool v0 = v0_in && ((((float)p->input_width)-2*(float)Cs[warpid*64+laneid])
            //                 >=(p->bn_gpu[by*8+gy]));
            // bool v1 = v1_in && ((((float)p->input_width)-2*(float)Cs[warpid*64+32+laneid])
            //                 >=(p->bn_gpu[by*8+gy]));

            union{ uin32 data; uin8 elements[4];} p0, p1;

            // pack into 32 1-bit.
            p0.data = __brev(__ballot_sync(0xFFFFFFFF, v0 > 0));
            p1.data = __brev(__ballot_sync(0xFFFFFFFF, v1 > 0));

            // printf("p0.data: %u, p1.data: %u\n", p0.data, p1.data); // ok, all 1s.
            __syncthreads();

            // output to binary after compression.
            if (laneid < 4)
            {
                Cb[(bx*8+laneid)*gdm*16+FLIPBITS(by,2)] = p0.elements[3-laneid]; 
                Cb[(bx*8+4+laneid)*gdm*16+FLIPBITS(by,2)] = p1.elements[3-laneid]; 
            }
        }
        //end
    }
}

__device__ __inline__ void Out128Layer(Out128LayerParam* p)
{
    using namespace nvcuda;
    GET_LANEID;
    GET_WARPID;
    using namespace nvcuda::wmma::experimental;
    extern __shared__ int Cs[];
    
    const int act_bit = p->act_bit;
    const int w_bit = p->w_bit;

    const int act_offset = PAD8(p->input_height)*STEP128(p->input_width)*128/32;
    const int w_offset = STEP128(p->weight_height)*PAD8(p->weight_width)*128/32;

    const int gdx = STEP8(p->input_height); //vertical
    const int gdy = STEP8(p->weight_width); //horizontal
    const int gdk = STEP128(p->input_width);

    // printf("act_bit: %d, w_bit: %d, act_offset: %d, w_offset: %d, gdx: %d, gdy: %d, gdk: %d\n", act_bit, w_bit, act_offset, w_offset, gdx, gdy, gdk);

    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy; bid+=gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> tmp_frag;

        const int bx = bid / gdy;
        const int by = bid % gdy;

        wmma::fill_fragment(c_frag, 0);

        for (int bit = 0; bit < act_bit*w_bit; bit++){

            int b_act = bit % act_bit;
            int b_w = bit / act_bit;
            int b_opt = b_act + b_w;

            // accmuluation of the current bit.
            wmma::fill_fragment(tmp_frag, 0);

            for (int i=0; i<gdk; i++)
            {
                load_matrix_sync(a_frag, p->input_gpu + b_act*act_offset + bx*8*gdk*4 + i*128/32, gdk*128);
                load_matrix_sync(b_frag, p->weight_gpu + b_w*w_offset + by*8*gdk*4 + i*128/32, gdk*128);
                // printf("weight: %u\n",  p->weight_gpu[0]);
                // printf("input: %u\n",  p->input_gpu[0]);
                bmma_sync(tmp_frag, a_frag, b_frag, tmp_frag, bmmaBitOpAND);
            }

            // Accumulation.
            #pragma unroll
            for (int t = 0; t < tmp_frag.num_elements; t++) 
            {
                // printf("%d\n", c_frag.x[t]);
                c_frag.x[t] += (tmp_frag.x[t] << b_opt);
            }
            __syncwarp();
        }

        store_matrix_sync(&Cs[warpid*64], c_frag, 8, wmma::mem_row_major);
        // if (laneid == 0 && warpid == 0){
        //     for (int i=0; i<8; i++){
        //         for (int j=0; j<8; j++){
        //             printf("%u ", Cs[warpid*64 + i * 8 + j]);
        //         }
        //         printf("\n");
        //     }
        // }

        uin32* output_sub = &(p->output_gpu[bx*(p->weight_width)*8+by*8]);

        if (laneid < 8)
        {
            for (int j=0; j<8; j++)
            {
                if ((bx*8+j)<(p->input_height))
                {
                    if (by*8+laneid<(p->weight_width))
                    {
                        // uin32 val = ((uin32)(p->input_width) 
                        //         - ((uin32)Cs[warpid*64+j*8+laneid])*2.0f)*
                        //             (p->bn_scale_gpu[by*8+laneid]) 
                        //         + (p->bn_bias_gpu[by*8+laneid]);

                        uin32 val = (uin32) Cs[warpid*64+j*8+laneid]; //* (p->bn_scale_gpu[by*8+laneid]) + (p->bn_bias_gpu[by*8+laneid]);
                        output_sub[j*(p->weight_width)+laneid] = val;
                        // printf("lanid: %d, val: %u\n", laneid, val);
                    }
                }
            }
        } //end
    }
}

//=================================== FMT ================================
__device__ __inline__ void In128LayerFMT(In128LayerParam* p)
{
    GET_LANEID;
    GET_WARPID;
    const int gdx = STEP8(p->input_height);
    const int gdy = STEP128(p->input_width);
    const int lx = (warpid>>2);
    const int ly = (warpid&0x3);
    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {
        const int bx = bid / gdy;
        const int by = bid % gdy;
        float f0 = ( (by*128+ly*32+laneid<(p->input_width)) 
                &&   (bx*8+lx<(p->input_height)) )?
            p->input_gpu[(bx*8+lx)*(p->input_width)+by*128+ly*32+laneid]:-1.0f;
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>=0));
        //New format
        if (laneid==0) p->output_gpu[(bx*gdy+by)*32+warpid] = r0;
    }
}

__device__ __inline__ void Fc128LayerFMT(Fc128LayerParam* p)
{
    using namespace nvcuda;
    GET_LANEID;
    GET_WARPID;
    using namespace nvcuda::wmma::experimental;
    extern __shared__ int Cs[];

    const int gdx = STEP8(p->input_height); //vertical
    const int gdy = STEP8(p->weight_width); //horizontal
    const int gdk = STEP128(p->input_width);
    const int gdm = STEP128(p->weight_width);

    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy; bid+=gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;

        wmma::fill_fragment(c_frag, 0);
        const int bx = bid / gdy;
        const int by = bid % gdy;

        for (int i=0; i<gdk; i++)
        {
            load_matrix_sync(a_frag, p->input_gpu + bx*8*gdk*4 + i*128*8/32, 128);
            load_matrix_sync(b_frag, p->weight_gpu + by*8*gdk*4 + i*128*8/32, 128);
            bmma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        store_matrix_sync(&Cs[warpid*64], c_frag, 8, wmma::mem_row_major);
        uin8* Cb = (uin8*)(&(p->output_gpu[0])); 

        const int gy = (laneid%8);
        const int gx = (laneid/8);
        bool v0_in = ((by*8+gy)<(p->output_width)) && ((bx*8+gx)<(p->output_height));
        bool v1_in = ((by*8+gy)<(p->output_width)) && ((bx*8+gx+4)<(p->output_height)); 
        bool v0 = v0_in && ((((float)p->input_width)-2*(float)Cs[warpid*64+laneid])
                        >=(p->bn_gpu[by*8+gy]));
        bool v1 = v1_in && ((((float)p->input_width)-2*(float)Cs[warpid*64+32+laneid])
                        >=(p->bn_gpu[by*8+gy]));

        union{ uin32 data; uin8 elements[4];} p0, p1;
        p0.data = __brev(__ballot_sync(0xFFFFFFFF, v0 ));
        p1.data = __brev(__ballot_sync(0xFFFFFFFF, v1 ));
        __syncthreads();

        if (laneid < 4)
        {
            Cb[(bx*gdm + by/16)*128+ laneid*16 + FLIPBITS((by%16),2)] = p0.elements[3-laneid];
            Cb[(bx*gdm + by/16)*128+ (laneid+4)*16 + FLIPBITS((by%16),2)] = p1.elements[3-laneid];
        }
    }
}



__device__ __inline__ void Out128LayerFMT(Out128LayerParam* p)
{
    using namespace nvcuda;
    GET_LANEID;
    GET_WARPID;
    using namespace nvcuda::wmma::experimental;
    extern __shared__ int Cs[];
    
    const int gdx = STEP8(p->input_height); //vertical
    const int gdy = STEP8(p->weight_width); //horizontal
    const int gdk = STEP128(p->input_width);

    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy; bid+=gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;

        wmma::fill_fragment(c_frag, 0);
        const int bx = bid / gdy;
        const int by = bid % gdy;

        for (int i=0; i<gdk; i++)
        {
            load_matrix_sync(a_frag, p->input_gpu + bx*8*gdk*4 + i*128*8/32, 128);
            load_matrix_sync(b_frag, p->weight_gpu + by*8*gdk*4 + i*128*8/32, 128);

            bmma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        store_matrix_sync(&Cs[warpid*64], c_frag, 8, wmma::mem_row_major);
        uin32* output_sub = &(p->output_gpu[bx*(p->weight_width)*8+by*8]);

        if (laneid < 8)
        {
            for (int j=0; j<8; j++)
            {
                if ((bx*8+j)<(p->input_height))
                {
                    if (by*8+laneid<(p->weight_width))
                    {
                        // output_sub[j*(p->weight_width)+laneid] = ((float)(p->input_width) 
                        //         - (float)Cs[warpid*64+j*8+laneid]*2.0f)*
                        //             (p->bn_scale_gpu[by*8+laneid]) 
                        //         + (p->bn_bias_gpu[by*8+laneid]);
                        output_sub[j*(p->weight_width)+laneid] = Cs[warpid*64+j*8+laneid] * (p->bn_scale_gpu[by*8+laneid]) + (p->bn_bias_gpu[by*8+laneid]);
                    }
                }
            }
        }
        //end
    }
}


//================================ Convolution ====================================
__device__ __inline__ void InConv128Layer(InConv128LayerParam* p)
{
    GET_LANEID;
    GET_WARPID;

    int act_bit = p->act_bit;
    int w_bit  = p->w_bit;

    // filter dimension [K, K, O, C]
    const int filter_offset = STEP32((p->filter_height)*(p->filter_width)*PAD128(p->output_channels)*(p->input_channels));

    extern __shared__ int Cs[];
    const int ots = STEP32(p->output_channels); //number of steps in K: output_channels
    const int otm = STEP128(p->output_channels);

    // each warp manage one output channels for all elements in that channel.
    // Cs == num_warp * output_channel * (float/uin32).
    volatile float* Csub = (float*)&Cs[warpid*(p->output_channels)]; 

    // filter is placed right after the Csub
    volatile uin32* s_filter = (uin32*)&Cs[32*(p->output_channels)]; 

    const int src_output_height = (p->pool_height)*(p->output_height); // input layer  (p->pool_height) == 1
    const int src_output_width = (p->pool_width)*(p->output_width); // input layer (p->pool_width) == 1

    // cache all filter elements in the shared memory.
    #pragma unroll
    for (int i=threadIdx.x; i < w_bit*(p->filter_height)*(p->filter_width)*(p->input_channels)*ots; i+=32*32){
        s_filter[i] = p->filter_gpu[i];
    }

    __syncthreads();

    //process by warp
    //iterate all output elements by warps
    // one "virtual" elements in the output across all output channels.
    // i.e. (x,y,c_out), where c_out is the iteration direction for one warp
    for (int bid = blockIdx.x*32+warpid; bid < src_output_height*src_output_width*(p->batch); bid += gridDim.x*32)
    {
        // Get the position in the output feature map. (N, H, W) = (bz, by, bx)
        const int bz = bid/(src_output_width*src_output_height); //over N: batch item id
        const int by = (bid%(src_output_width*src_output_height))/(src_output_width);//over P:out_height
        const int bx = (bid%(src_output_width*src_output_height))%(src_output_width);//over Q:out_width 

        //coord (ax,ay) in Input from bx,by in Output
        const int ax0 = bx*(p->stride_width)-(p->pad_w);
        const int ay0 = by*(p->stride_height)-(p->pad_h);

        // initialize the intra warp results to zero. Csub is per warp.
        for (int i=laneid; i<(p->output_channels); i+=32) Csub[i] = 0;

        //load a window of data from Input
        for (int r=0; r<(p->filter_height); r++)
        {
            const int ay = ay0 + r;  //y-coord in Input
            if ((ay>=0) && (ay<(p->input_height))) // check the boundry
            {
                for (int s=0; s<(p->filter_width); s++)
                {
                    const int ax = ax0 + s; //x-coord in Input

                    //within Input frame
                    if ((ax>=0) && (ax<(p->input_width)) )
                    {
                        // load the RGB channel, 
                        // float per channel.
                        // same (x, y) position across all channels.
                        // [N, C, H, W]
                        float f0 = p->input_gpu[(bz*3+0)*(p->input_height)*(p->input_width) + ay*(p->input_width)+ax];//R
                        float f1 = p->input_gpu[(bz*3+1)*(p->input_height)*(p->input_width) + ay*(p->input_width)+ax];//G
                        float f2 = p->input_gpu[(bz*3+2)*(p->input_height)*(p->input_width) + ay*(p->input_width)+ax];//B
                        
                        // if (ay == ax && ax == 0){
                        //     printf("input@[0, 0], R: %.3f, G:%.3f, B:%.3f\n", f0, f1, f2);
                        // }

                        // output channel was compressed 1-bit to 32-bit
                        // now recovered back.
                        // [H, W, C, O]
                        for (int k=0; k<ots; k++)
                        {
                            for (int bIdx = 0; bIdx < w_bit; bIdx++){
                                uin32 l0 = s_filter[bIdx*filter_offset + (r*(p->filter_width)+s)*(p->input_channels)*ots + 0*ots+k];
                                uin32 l1 = s_filter[bIdx*filter_offset + (r*(p->filter_width)+s)*(p->input_channels)*ots + 1*ots+k];
                                uin32 l2 = s_filter[bIdx*filter_offset + (r*(p->filter_width)+s)*(p->input_channels)*ots + 2*ots+k];
                                
                                // uin32 l0 = p->filter_gpu[bIdx*filter_offset + (r*(p->filter_width)+s)*(p->input_channels)*ots + 0*ots+k];
                                // uin32 l1 = p->filter_gpu[bIdx*filter_offset + (r*(p->filter_width)+s)*(p->input_channels)*ots + 1*ots+k];
                                // uin32 l2 = p->filter_gpu[bIdx*filter_offset + (r*(p->filter_width)+s)*(p->input_channels)*ots + 2*ots+k];

                                // if (r == s && s == 0 && k == 0 && laneid==0 && bIdx == 1){
                                //      printf("filter, l0: %u, l1:%u, l2:%u\n", (l0>>31)&0x1 , (l1>>31)&0x1, (l2>>31)&0x1);
                                // }
                                
                                Csub[32*k+laneid] += ( (((l0>>(31-laneid))&0x1)*f0) + (((l1>>(31-laneid))&0x1)*f1) + (((l2>>(31-laneid))&0x1)*f2) ) * (1 << bIdx);
                                
                                // if (r == s && s == 0 && k == 0 && laneid==0 && bIdx == 1){
                                //     printf("Csub: %.3f\n", Csub[32*k+laneid]);
                                // }
                                // Csub[32*k+laneid] += (((l0>>(31-laneid))&0x1)?f0:-f0)
                                // + (((l1>>(31-laneid))&0x1)?f1:-f1)
                                // + (((l2>>(31-laneid))&0x1)?f2:-f2);
                            }
                        } // END for (k)
                    } // END if ((ax>=0) && (ax<(p->input_width)) )
                } // END for (s)
            } // END if ((ay>=0) && (ay<(p->input_height))) // check the boundry
        } //END for (r) 

        // To shape[batch, input_height, input_width, in_channels/32]
        const int dst_y = by/(p->pool_height);
        const int dst_x = bx/(p->pool_width);

        //const int idx = (bz*(p->output_height)*(p->output_width) //N
        //+dst_y*(p->output_width) //P
        //+dst_x)*ots; //Q

        const int output_offset = (p->output_height)*(p->output_width)*PAD8(p->batch)*STEP32(p->output_channels);

        // To shape[input_height, input_width, batch, in_channels/32]
        // [H, W, N, C]
        // const int idx = (dst_y*(p->output_width)*PAD8(p->batch) //P
        //                 +dst_x*PAD8(p->batch) //Q
        //                 +bz)*otm*4;

        //* new CONV format [N, H, W, C]
        const int idx = bz*(p->output_width)*(p->output_height)*otm*4   // N
                        + dst_y*(p->output_width)*otm*4                // H
                        + dst_x*otm*4;                                  // W

        // const int idx = (dst_y*(p->output_width)*PAD8(p->batch) //P
        //                 +dst_x*PAD8(p->batch) //Q
        //                 +bz)*otm*4;

        for (int k=0; k<ots; k++)
        {
            // save shape [batch, output_height, output_width, out_channels/64]
            // bool bin = (float)(Csub[k*32+laneid])<(p->bn_gpu)[k*32+laneid]?0:1;
            uin32 qnt_val = quantize(Csub[k*32+laneid], act_bit);

            // if (bz == bx && bx == by && bz == 0 && laneid == 0 && k == 0){
                // uin32 test = quantize(Csub[k*32+laneid], act_bit);
                // printf("act_bit: %u\n", act_bit);
            // }
            
            #pragma unroll
            for (int bIdx = 0; bIdx < act_bit; bIdx++){
                uin32 C = __brev(__ballot_sync(0xFFFFFFFF, ((qnt_val>>bIdx) & 0x1) > 0));
                if (laneid==0) {
                    // p->output_gpu[bIdx*output_offset + idx + k] = C;
                    atomicOr(&p->output_gpu[bIdx*output_offset + idx + k], C); //Q
                }
            }
        } // end for (k)


        // for residual network
        if (p->save_residual)
        {
            for (int k=0; k<ots; k++)
            {
                p->output_residual_gpu[(by*(p->output_width)+bx)*PAD8(p->batch)
                    *PAD128(p->output_channels) + bz*PAD128(p->output_channels)
                    + k*32 + laneid] = Csub[k*32+laneid];
            }
        } // END if (residual)

    } // END for (bid)
}

__device__ __inline__ void Conv128Layer(Conv128LayerParam* p)
{
    using namespace nvcuda;
    using namespace nvcuda::wmma::experimental;
    GET_LANEID;
    GET_WARPID;

    // arbitary precision support variables.
    const int w_bit = p->w_bit;
    const int act_bit = p->act_bit;

    // input bit offset by uin32 [H, W, N, C]
    const int in_offset = STEP32((p->input_height)*(p->input_width)*(p->batch)*(p->input_channels)); 
    // weight bit offset by uin32 [K, K, O, C]
    const int w_offset = STEP32((p->filter_height)*(p->filter_width)*(p->output_channels)*(p->input_channels));
    // output bit offset by uin32 [H, W, N, C]
    const int opt_offset = STEP32((p->output_height)*(p->output_width)*(p->batch)*(p->output_channels));

    const int ins = STEP128(p->input_channels); //1
    const int ots = STEP32(p->output_channels); //4
    const int bas = STEP8(p->batch);//4

    const int src_output_height = (p->pool_height)*(p->output_height);//32
    const int src_output_width = (p->pool_width)*(p->output_width);//32
    
    // each warp manage one output.
    extern __shared__ int Cs[];

    // each block consists of 32x32 threads, 32 warps
    // each warp manage 8 points in the output feature map.
    // /32 * /8 =  /256 = /8*8*4
    for (int bid = blockIdx.x*32+warpid; bid < src_output_height*src_output_width*ots*bas; bid += gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b0_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b1_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b2_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b3_frag;

        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c0_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c1_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c2_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c3_frag;

        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c0_frag_tmp;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c1_frag_tmp;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c2_frag_tmp;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c3_frag_tmp;

        // get the position of a point in the output feature map
        // each warp compute 4 x 8 x 8 output region of one output channel.
        // (by, bx, bz, bo) = (H, W, N, C)
        const int by = bid/(src_output_width*ots*bas);              //P: output_height
        const int bx = (bid%(src_output_width*ots*bas))/(ots*bas); //Q:output_width
        const int bz = (bid%(src_output_width*ots*bas))%(ots*bas); //output_channel/32*batch/8
        const int bn = bz / ots; //N:batch (8)
        const int bo = bz % ots; //O:out_channel (4*8)

        //coord (ax,ay) in Input from bx,by in Output
        const int ax0 = bx*(p->stride_width)-(p->pad_w);
        const int ay0 = by*(p->stride_height)-(p->pad_h);

        //track the number of filter entries that are masked off
        int exclude = 0;
        wmma::fill_fragment(c0_frag, 0);
        wmma::fill_fragment(c1_frag, 0);
        wmma::fill_fragment(c2_frag, 0);
        wmma::fill_fragment(c3_frag, 0);

        wmma::fill_fragment(c0_frag_tmp, 0);
        wmma::fill_fragment(c1_frag_tmp, 0);
        wmma::fill_fragment(c2_frag_tmp, 0);
        wmma::fill_fragment(c3_frag_tmp, 0);    

        //load a window of data from Input
        // vertical direction do convolution.
        // aggregation by x, y and input channel
        for (int r=0; r<(p->filter_height); r++)
        {
            const int ay = ay0 + r; //y-coord in Input
            for (int s=0; s<(p->filter_width); s++)
            {
                const int ax = ax0 + s; //x-coord in Input
                if ((ay>=0)&&(ay<(p->input_height))&&(ax>=0)&&(ax<(p->input_width)))
                {
                    // iterate along the input channel 128 element per step,
                    for (int c=0; c<ins; c++)
                    {
                        //input: [H,W,N,C], filter: [K,K,O,C]
                        // const int ins = STEP128(p->input_channels); //1 * (128/4)
                        // const int ots = STEP32(p->output_channels); //4
                        // const int bas = STEP8(p->batch);//4
                        // bn: item-id in a batch
                        // bo: output channel //O:out_channel (4*8)

                        for (int bIdx = 0; bIdx < act_bit*w_bit; bIdx++){
                            int b_act = bIdx % act_bit;
                            int b_w = bIdx / act_bit;
                            int b_shift = b_act + b_w;
                            
                            // a_frag using 1-bit for input --> input: [H,W,N,C]
                            load_matrix_sync(a_frag, &(p->input_gpu[b_act*in_offset + (ay*(p->input_width)+ax)*bas*8*ins*4 + 8*bn*ins*4 + c*4]), ins*128);

                            // b_frag using 1-bit for filter --> filter: [K,K,O,C]
                            load_matrix_sync(b0_frag, &(p->filter_gpu[b_w*w_offset + (r*(p->filter_width)+s)*ots*32*ins*4 + (bo*32+0)*ins*4 + c*4]), ins*128);
                            bmma_sync(c0_frag_tmp, a_frag, b0_frag, c0_frag_tmp, bmmaBitOpAND);

                            load_matrix_sync(b1_frag, &(p->filter_gpu[b_w*w_offset + (r*(p->filter_width)+s)*ots*32*ins*4 + (bo*32+8)*ins*4 + c*4]), ins*128);
                            bmma_sync(c1_frag_tmp, a_frag, b1_frag, c1_frag_tmp, bmmaBitOpAND);

                            load_matrix_sync(b2_frag, &(p->filter_gpu[b_w*w_offset + (r*(p->filter_width)+s)*ots*32*ins*4 + (bo*32+16)*ins*4 + c*4]), ins*128);
                            bmma_sync(c2_frag_tmp, a_frag, b2_frag, c2_frag_tmp, bmmaBitOpAND);

                            load_matrix_sync(b3_frag, &(p->filter_gpu[b_w*w_offset + (r*(p->filter_width)+s)*ots*32*ins*4 + (bo*32+24)*ins*4 + c*4]), ins*128);
                            bmma_sync(c3_frag_tmp, a_frag, b3_frag, c3_frag_tmp, bmmaBitOpAND);

                            #pragma unroll
                            for (int i=0; i < c0_frag.num_elements; i++){
                                c0_frag.x[i] += (c0_frag_tmp.x[i]<<b_shift);
                                c1_frag.x[i] += (c1_frag_tmp.x[i]<<b_shift);
                                c2_frag.x[i] += (c2_frag_tmp.x[i]<<b_shift);
                                c3_frag.x[i] += (c3_frag_tmp.x[i]<<b_shift);
                                // if (b_w == 0){
                                //     printf("c0: %u\n", (c0_frag_tmp.x[i]));
                                //     printf("c1: %u\n", (c1_frag_tmp.x[i]));
                                //     printf("c2: %u\n", (c2_frag_tmp.x[i]));
                                //     printf("c3: %u\n", (c3_frag_tmp.x[i]));
                                // }
                            }

                            wmma::fill_fragment(c0_frag_tmp, 0);
                            wmma::fill_fragment(c1_frag_tmp, 0);
                            wmma::fill_fragment(c2_frag_tmp, 0);
                            wmma::fill_fragment(c3_frag_tmp, 0);   
                        }
                    } // END for (c)
                } // END for ()
                else //not in frame
                {
                    exclude++; //accumulate
                } // <-- END: if ((ay>=0)&&(ay<(p->input_height))&&(ax>=0)&&(ax<(p->input_width)))
            } // <-- END: for (s)
        } // <--END: for (r)

        // 256 = 4 x 8 x 8
        // 4 x (8 x 8) aligned horizontially leading to the stride of 32.
        store_matrix_sync(&Cs[warpid*256+0],  c0_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+8],  c1_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+16], c2_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+24], c3_frag, 32, wmma::mem_row_major);

        __syncthreads();

        // process 8 x (8 x 4) line by line
        for (int b=0; b<8; b++)
        {
            // get the current element from 1 x (8 x 4) line.
            // int res = (int)(p->input_channels)*(p->filter_width)*(p->filter_height) //C*R*S
            //     - (int)exclude*(p->input_channels) //eliminate padding distoration 
            //     - (int)(2*Cs[warpid*256+b*32+laneid]);//n-2acc(a^b) for 0/1 to sim +1/-1

            uin32 res = Cs[warpid*256+b*32+laneid];
            uin32 res_qnt = quantize(res, act_bit);

            // if (bn == 0 && b == 0 && by == 0 && bx == 1 && bo == 0 && laneid == 0){
            //     uin32 test = quantize(res, act_bit);
            //     printf("=> by: %d, bx: %d, test: %u\n", by, bx, res);
            //     printf("=> by: %d, bx: %d, test_qnt: %u\n", by, bx, test);
            // }

            // if (bn == 0 && b == 0 && by < 3 && bx < 3 && bo == 0 && laneid == 0){
            //     uin32 test = quantize(res, act_bit);
            //     printf("=> by: %d, bx: %d, test: %u\n", by, bx, res);
                // printf("=> by: %d, bx: %d, test_qnt: %u\n", by, bx, test);
            // }

            // if (p->inject_residual && ((bo*32+laneid)<(p->residual_channels)))
            // {
            //     int residual = 0;
            //     if (p->residual_pool)
            //     {
            //     /*

            //         //if((bn*8+b)<(p->batch) && (bo*32+laneid)<(p->residual_channels))
            //         {
            //         int pl0 = p->input_residual_gpu[(by*2+0)*2*(p->output_width)*
            //                 bas*8*PAD128(p->residual_channels)
            //                 +(bx*2+0)*bas*8*PAD128(p->residual_channels)
            //                 +(bn*8+b)*PAD128(p->residual_channels)+bo*32+laneid];

            //         int pl1 = p->input_residual_gpu[(by*2+0)*2*(p->output_width)*
            //                 bas*8*PAD128(p->residual_channels)
            //                 +(bx*2+1)*bas*8*PAD128(p->residual_channels)
            //                 +(bn*8+b)*PAD128(p->residual_channels)+bo*32+laneid];

            //         int pl2 = p->input_residual_gpu[(by*2+1)*2*(p->output_width)*
            //                 bas*8*PAD128(p->residual_channels)
            //                 +(bx*2+0)*bas*8*PAD128(p->residual_channels)
            //                 +(bn*8+b)*PAD128(p->residual_channels)+bo*32+laneid];

            //         int pl3 = p->input_residual_gpu[(by*2+1)*2*(p->output_width)*
            //                 bas*8*PAD128(p->residual_channels)
            //                 +(bx*2+1)*bas*8*PAD128(p->residual_channels)
            //                 +(bn*8+b)*PAD128(p->residual_channels)+bo*32+laneid];
            //         residual = max(pl3,max(pl2,max(pl0,pl1)));
            //         }
            //         */

            //         residual = INT_MIN;
            //         for (int i=0; i<2; i++)
            //             for( int j=0; j<2; j++)
            //                 residual = max(residual, p->input_residual_gpu[
            //                         (by*2+i)*2*(p->output_width)*
            //                         bas*8*PAD128(p->residual_channels)
            //                         +(bx*2+j)*bas*8*PAD128(p->residual_channels)
            //                         +(bn*8+b)*PAD128(p->residual_channels)+bo*32+laneid]);

            //     }
            //     else
            //     {
            //         //residual = p->input_residual_gpu[by*(p->output_width)
            //         residual = p->input_residual_gpu[by*src_output_width
            //             *bas*8*PAD128(p->residual_channels)
            //             +bx*bas*8*PAD128(p->residual_channels)
            //             +(bn*8+b)*PAD128(p->residual_channels)
            //             +bo*32+laneid];
            //     }
            //     res += residual;
            // }

            // compress the 1 x (8 x 4) bits -->  1 32-bit unsigned.
            // unsigned C = __brev(__ballot_sync( 0xFFFFFFFF, 
            //             (float)res<(p->bn_gpu[bo*32+laneid])?0:1));

            // if before a FC layer.
            if (p->ahead_fc)
            {
                // max = a ^ ((a ^ max) & -(a < max));

                // if (by%(p->pool_height) == 0 && bx%(p->pool_width) == 0){
                //     int opt_idy = by/(p->pool_height);
                //     int opt_idx = bx/(p->pool_width);
                //     for (int h_idx = by; h_idx < by+(p->pool_height); h_idx++){
                //         for (int w_idx = bx; w_idx < bx+(p->pool_width); w_idx++){
                            
                //         }
                //     }
                #pragma unroll
                for (int bIdx = 0; bIdx < act_bit; bIdx++){
                    uin32 tmp = __brev(__ballot_sync( 0xFFFFFFFF, ((res_qnt>>bIdx)&0x1)>0));
                    if (laneid==0) //For FC layer [N, H, W, C]
                        atomicOr(&(p->output_gpu[bIdx*opt_offset 
                            + (bn*8+b)*(p->output_height)*(p->output_width)*STEP128(p->output_channels)*4
                            + (by/(p->pool_height))*(p->output_width)*STEP128(p->output_channels)*4
                            + (bx/(p->pool_width))*STEP128(p->output_channels)*4
                            + bo]), tmp);
                }
                // }

                //atomicOr(&p->output_gpu[((by/(p->pool_height))*(p->output_width)
                //*bas*8*STEP128(p->output_channels)*4) //P
                //+ ((bx/(p->pool_width))*bas*8*STEP128(p->output_channels)*4) //Q
                //+ bo*bas*8 + (bn*8+b)],C);
            }
            else
            {

                // int opt_idx = (by/(p->pool_height))*(p->output_width)*(p->batch)*(p->output_channels)
                //             + (bx/(p->pool_width))*(p->batch)*(p->output_channels)
                //             + (bn*8+b)*(p->output_channels) 
                //             + bo;

                // atomic_max(output_tmp[opt_idx], res_qnt);
                #pragma unroll
                for (int bIdx = 0; bIdx < act_bit; bIdx++){
                    uin32 tmp = __brev(__ballot_sync( 0xFFFFFFFF, ((res_qnt>>bIdx)&0x1)>0));
                    if (laneid==0) //For normal convolution layer [H, W, N, C]
                        atomicOr(&p->output_gpu[bIdx*opt_offset 
                            + (by/(p->pool_height))*(p->output_width)*bas*8*STEP128(p->output_channels)*4 //P
                            + (bx/(p->pool_width))*bas*8*STEP128(p->output_channels)*4 //Q
                            + (bn*8+b)*STEP128(p->output_channels)*4 
                            + bo], tmp); // output_tmp[opt_idx]); 
                            // tmp);
                    }
            }

            if (p->save_residual)
            {
                p->output_residual_gpu[by*(p->output_width)*bas*8*PAD128(p->output_channels)
                    + bx*bas*8*PAD128(p->output_channels)
                    + (bn*8+b)*PAD128(p->output_channels) 
                    + bo*32 + laneid] = res;
            }
        } // END for (b)
    } // END for (bid)
}

//================================ Convolution FMT ====================================

__device__ __inline__ void InConv128LayerFMT(InConv128LayerParam* p)
{
    GET_LANEID;
    GET_WARPID;
    extern __shared__ int Cs[];
    const int ots = STEP32(p->output_channels); //number of steps in K: output_channels
    const int otm = STEP128(p->output_channels);
    volatile float* Csub = (float*)&Cs[warpid*(p->output_channels)];
    volatile uin32* s_filter = (uin32*)&Cs[32*(p->output_channels)]; 
    const int src_output_height = (p->pool_height)*(p->output_height);
    const int src_output_width = (p->pool_width)*(p->output_width);

    for (int i=threadIdx.x; i<(p->filter_height)*(p->filter_width)
            *(p->input_channels)*ots; i+=32*32) 
        s_filter[i] = p->filter_gpu[i];
    __syncthreads();

    for (int bid = blockIdx.x*32+warpid; bid < src_output_height*src_output_width*(p->batch);
            bid += gridDim.x*32)
    {
        const int bz = bid/(src_output_width*src_output_height); //over N:batch
        const int by = (bid%(src_output_width*src_output_height))
            /(src_output_width);//over P:out_height
        const int bx = (bid%(src_output_width*src_output_height))
            %(src_output_width);//over Q:out_width 
        //coord (ax,ay) in Input from bx,by in Output
        const int ax0 = bx*(p->stride_width)-(p->pad_w);
        const int ay0 = by*(p->stride_height)-(p->pad_h);
        for (int i=laneid; i<(p->output_channels); i+=32) Csub[i] = 0;

        //load a window of data from Input
        for (int r=0; r<(p->filter_height); r++)
        {
            const int ay = ay0 + r; //y-coord in Input
            if ((ay>=0) && (ay<(p->input_height))) 
            {
                for (int s=0; s<(p->filter_width); s++)
                {
                    const int ax = ax0 + s; //x-coord in Input
                    //within Input frame
                    if ((ax>=0) && (ax<(p->input_width)) )
                    {
                        float f0 = p->input_gpu[(bz*3+0)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//R
                        float f1 = p->input_gpu[(bz*3+1)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//G
                        float f2 = p->input_gpu[(bz*3+2)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//B

                        for (int k=0; k<ots; k++)
                        {
                            uin32 l0 = s_filter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 0*ots+k];
                            uin32 l1 = s_filter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 1*ots+k];
                            uin32 l2 = s_filter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 2*ots+k];

                            Csub[32*k+laneid] += (((l0>>(31-laneid))&0x1)?f0:-f0)
                                + (((l1>>(31-laneid))&0x1)?f1:-f1)
                                + (((l2>>(31-laneid))&0x1)?f2:-f2);
                        }
                    }
                }
            }
        }

        // To shape[batch, input_height, input_width, in_channels/32]
        const int dst_y = by/(p->pool_height);
        const int dst_x = bx/(p->pool_width);

        // To shape[input_height, input_width, batch/8*in_channels/128, batch8*in_channels128/32]
        const int idx = dst_y*(p->output_width)*PAD8(p->batch)*otm*4 //P
                +dst_x*PAD8(p->batch)*otm*4; //Q
        for (int k=0; k<ots; k++)
        {
            // save shape[batch, output_height, output_width, out_channels/64]
            bool bin = (float)(Csub[k*32+laneid])<(p->bn_gpu)[k*32+laneid]?0:1;
            unsigned C = __brev(__ballot_sync(0xFFFFFFFF, bin));

            //if (laneid==0) atomicOr(&p->output_gpu[idx+bz*otm*4+k], C); //Q
            if (laneid==0) atomicOr(&p->output_gpu[idx
                    +((bz/8)*otm+k/4)*32+((bz%8)*4+k%4)], C); //Q
        }
        if (p->save_residual)
        {
            for (int k=0; k<ots; k++)
            {
                p->output_residual_gpu[(by*(p->output_width)+bx)*PAD8(p->batch)
                    *PAD128(p->output_channels) + bz*PAD128(p->output_channels)
                    + k*32 + laneid] = Csub[k*32+laneid];
            }
        }
    }
}



__device__ __inline__ void Conv128LayerFMT(Conv128LayerParam* p)
{
    using namespace nvcuda;
    using namespace nvcuda::wmma::experimental;
    GET_LANEID;
    GET_WARPID;
    const int ins = STEP128(p->input_channels); //1
    const int ots = STEP32(p->output_channels); //4
    const int bas = STEP8(p->batch);//4
    const int src_output_height = (p->pool_height)*(p->output_height);//32
    const int src_output_width = (p->pool_width)*(p->output_width);//32
    extern __shared__ int Cs[];
    volatile int* Csub = (int*)Cs[warpid];

    for (int bid = blockIdx.x*32+warpid; bid < src_output_height*src_output_width*ots*bas;
            bid += gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b0_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b1_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b2_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b3_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c0_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c1_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c2_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c3_frag;

        const int by = bid/(src_output_width*ots*bas); //P: output_height
        const int bx = (bid%(src_output_width*ots*bas))/(ots*bas); //Q:output_width
        const int bz = (bid%(src_output_width*ots*bas))%(ots*bas); //output_channel/32*batch/8
        const int bn = bz / ots; //N:batch (8)
        const int bo = bz % ots; //O:out_channel (4*8)

        //coord (ax,ay) in Input from bx,by in Output
        const int ax0 = bx*(p->stride_width)-(p->pad_w);
        const int ay0 = by*(p->stride_height)-(p->pad_h);
        //track the number of filter entries that are masked off
        int exclude = 0;
        wmma::fill_fragment(c0_frag, 0);
        wmma::fill_fragment(c1_frag, 0);
        wmma::fill_fragment(c2_frag, 0);
        wmma::fill_fragment(c3_frag, 0);

        //load a window of data from Input
        for (int r=0; r<(p->filter_height); r++)
        {
            const int ay = ay0 + r; //y-coord in Input
            for (int s=0; s<(p->filter_width); s++)
            {
                const int ax = ax0 + s; //x-coord in Input
                if ((ay>=0)&&(ay<(p->input_height))&&(ax>=0)&&(ax<(p->input_width)))
                {
                    for (int c=0; c<ins; c++)
                    {
                        //input: [H,W,N,C], filter: [K,K,O,C]
                        load_matrix_sync(a_frag, 
                            &(p->input_gpu[(ay*(p->input_width)+ax)*bas*8*ins*4
                            +8*bn*ins*4+c*4*8]), 128);
                        load_matrix_sync(b0_frag, 
                            &(p->filter_gpu[(r*(p->filter_width)+s)*ots*32*ins*4
                            +(bo*32+0)*ins*4+c*4*8]), 128);
                        bmma_sync(c0_frag, a_frag, b0_frag, c0_frag);
                        load_matrix_sync(b1_frag, 
                            &(p->filter_gpu[(r*(p->filter_width)+s)*ots*32*ins*4
                            +(bo*32+8)*ins*4+c*4*8]), 128);
                        bmma_sync(c1_frag, a_frag, b1_frag, c1_frag);
                        load_matrix_sync(b2_frag, 
                            &(p->filter_gpu[(r*(p->filter_width)+s)*ots*32*ins*4
                            +(bo*32+16)*ins*4+c*4*8]), 128);
                        bmma_sync(c2_frag, a_frag, b2_frag, c2_frag);
                        load_matrix_sync(b3_frag, 
                            &(p->filter_gpu[(r*(p->filter_width)+s)*ots*32*ins*4
                            +(bo*32+24)*ins*4+c*4*8]), 128);
                        bmma_sync(c3_frag, a_frag, b3_frag, c3_frag);
                    }
                }
                else //not in frame
                {
                    exclude++; //accumulate
                }
            }
        }
        store_matrix_sync(&Cs[warpid*256+0], c0_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+8], c1_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+16], c2_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+24], c3_frag, 32, wmma::mem_row_major);
        __syncthreads();

        for (int b=0; b<8; b++)
        {
            int res = (int)(p->input_channels)*(p->filter_width)*(p->filter_height) //C*R*S
                - (int)exclude*(p->input_channels) //eliminate padding distoration 
                - (int)(2*Cs[warpid*256+b*32+laneid]);//n-2acc(a^b) for 0/1 to sim +1/-1
            
            if (p->inject_residual && ((bo*32+laneid)<(p->residual_channels)))
            {
                int residual = 0;
                if (p->residual_pool)
                {
                    residual = INT_MIN;
                    for (int i=0; i<2; i++)
                        for( int j=0; j<2; j++)
                            residual = max(residual, p->input_residual_gpu[
                                    (by*2+i)*2*(p->output_width)*
                                    bas*8*PAD128(p->residual_channels)
                                    +(bx*2+j)*bas*8*PAD128(p->residual_channels)
                                    +(bn*8+b)*PAD128(p->residual_channels)+bo*32+laneid]);
                }
                else
                {
                    //residual = p->input_residual_gpu[by*(p->output_width)
                    residual = p->input_residual_gpu[by*src_output_width
                        *bas*8*PAD128(p->residual_channels)
                        +bx*bas*8*PAD128(p->residual_channels)
                        +(bn*8+b)*PAD128(p->residual_channels)
                        +bo*32+laneid];
                }
                res += residual;
            }
            unsigned C = __brev(__ballot_sync(0xFFFFFFFF,
                        (float)res<(p->bn_gpu[bo*32+laneid])?0:1));

            if (p->ahead_fc)
            {
                if (laneid==0)
                {
                    int otm = (p->output_height)*(p->output_width)*STEP128(p->output_channels);
                    int k = ((by/(p->pool_height))*(p->output_width)
                            *STEP128(p->output_channels)*4)
                        + ((bx/(p->pool_width))*STEP128(p->output_channels)*4) + bo;
                    atomicOr(&(p->output_gpu[(bn*otm+(k/4))*32+b*4+(k%4)]),C);
                }
            }
            else
            {
                if (laneid==0) //For normal convolution layer HWBC
                    atomicOr(&p->output_gpu[((by/(p->pool_height))*(p->output_width)
                                *bas*8*STEP128(p->output_channels)*4) //P
                            + ((bx/(p->pool_width))*bas*8*STEP128(p->output_channels)*4) //Q
                            + (bn*STEP128(p->output_channels)+(bo/4))*32 + b*4+(bo%4)],C);
            }

            if (p->save_residual)
            {
                p->output_residual_gpu[by*(p->output_width)*bas*8*PAD128(p->output_channels)
                    + bx*bas*8*PAD128(p->output_channels)
                    + (bn*8+b)*PAD128(p->output_channels) 
                    + bo*32 + laneid] = res;
            }
        }
    }
}

#endif
