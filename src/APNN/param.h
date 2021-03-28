// ---------------------------------------------------------------------------
// File: param.h
// Define basic layer objects.
// ---------------------------------------------------------------------------
// See our arXiv paper for detail: https://arxiv.org/abs/2006.16578
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/TCBNN
// PNNL-IPID: 31925-E, ECCN: EAR99, IR: PNNL-SA-152850
// BSD Lincese.
// Richland, 99352, WA, USA. June-30-2020.
// ---------------------------------------------------------------------------



#ifndef PARAM_H
#define PARAM_H

#include "utility.h"

#define max_v 10
#define min_v -10

// #define clip(x, lb, ub)                     \
// ({                                          \
//     if ((x) < (lb)) return (lb+1);            \
//     if ((x) > (ub)) return (ub-1);            \
//     return (x);                             \
// })  

__inline__ __device__ float clip(float x, float lb, float ub){
    if (x < lb) return lb+1;
    if (x > ub) return ub-1;
    return x;
}

__inline__ __device__ uin32 clip_int(uin32 x, uin32 lb, uin32 ub){
    if (x < lb) return lb+1;
    if (x > ub) return ub-1;
    return x;
}


const int dev=0;

// packing weight for the hidden FC layer. STEP128(A_height)*PAD128(A_width)
__global__ void PackFcWeight128(const uin32* __restrict__ A, uin32* B, 
                                    const int A_height, const int A_width, const int w_bit)
{
    GET_LANEID;
    GET_WARPID;

    const int gdx = STEP128(A_height);
    const int gdy = STEP8(A_width);

    const int lx = (warpid & 0x3); // warp x_index vertically
    const int ly = (warpid >> 2);  // warp y_index hozerionsally.

    const int offset = A_height*A_width;
    const int offset_opt = STEP128(A_height)*PAD128(A_width)*128/32;

    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {
        const int bx = bid % gdx;
        const int by = bid / gdx;
        
        for (int bIdx = 0; bIdx < w_bit; bIdx++){
            float f0 = ( (bx*128+lx*32+laneid<A_height) && (by*8+ly<A_width) )? A[bIdx*offset + (bx*128+lx*32+laneid)*A_width+by*8+ly]:-1.0f;
            unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0));
            if (laneid==0) 
            {
                // adaption for the new store format.
                // hozerontial y-axis offset: (by*8+ly)*gdx*4 --> w_bit*(by*8+ly)*gdx*4 + bIdx*gdx*4 
                // vertical x-axis offset: bx*4+lx
                // B[bIdx*offset_opt + (by*8+ly)*gdx*4 + bx*4 + lx] = r0;
                int pos = w_bit*(by*8+ly)*gdx*4 + bIdx*gdx*4  + bx*4+lx;
                B[pos] = r0;
            }
        }
    }
}


// packing weight for the output FC layer. STEP128(A_height)*PAD8(A_width)
__global__ void PackFcWeight128_OUTPUT(const uin32* __restrict__ A, uin32* B, 
                                    const int A_height, const int A_width, const int w_bit)
{
    GET_LANEID;
    GET_WARPID;

    const int gdx = STEP128(A_height);
    const int gdy = STEP8(A_width);

    const int lx = (warpid & 0x3); // warp x_index vertically
    const int ly = (warpid >> 2);  // warp y_index hozerionsally.

    const int offset = A_height*A_width;
    const int offset_opt = STEP128(A_height)*PAD8(A_width)*128/32;

    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {
        const int bx = bid % gdx;
        const int by = bid / gdx;
        
        for (int bIdx = 0; bIdx < w_bit; bIdx++){
            float f0 = ( (bx*128+lx*32+laneid<A_height) && (by*8+ly<A_width) )? A[bIdx*offset + (bx*128+lx*32+laneid)*A_width+by*8+ly]:-1.0f;
            unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0));
            if (laneid==0){
                // format for new kernel.
                // x-axis (by*8+ly)*gdx*4 --> w_bit*(by*8+ly)*gdx*4 + bIdx*gdx*4
                // y-axis bx*4+lx
                // B[bIdx*offset_opt + (by*8+ly)*gdx*4+ bx*4+lx] = r0;
                int pos = w_bit*(by*8+ly)*gdx*4 + bIdx*gdx*4 + bx*4+lx;
                B[pos] = r0;
            }
        }
    }
}


// from compressed bit feature map (bit, M/32, N) --> (M, N) in uin32
__global__ void UnPackFcWeight128(const uin32* __restrict__ A, uin32* B, 
                                  const int A_height, const int A_width, const int bitwidth)
{
    GET_LANEID;
    GET_WARPID;

    const int gdx = STEP128(A_height);
    const int gdy = STEP8(A_width);

    const int lx = (warpid & 0x3); // warp x_index vertical
    const int ly = (warpid >> 2);  // warp y_index horizontal

    const int offset_input = STEP128(A_height)*PAD128(A_width)*128/32;   // offset of input.

    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {
        const int bx = bid % gdx;
        const int by = bid / gdx;

        for (int bIdx = 0; bIdx < bitwidth; bIdx++){
            unsigned r0 = A[bIdx*offset_input + (by*8+ly)*gdx*4 + bx*4 + lx];
            // unsigned r0 = A[(bx*8+lx)*gdy*4+by*4+ly];
            if ((bx*128+lx*32+laneid<A_height) && (by*8+ly<A_width)){
                // B[bIdx * offset + (bx*8+lx)*A_width+by*128+ly*32+laneid] = 2*(uin32)((r0>>(31-laneid)) & 0x1) - 1; 
                B[(bx*128+lx*32+laneid)*A_width + by*8 + ly] += (uin32)((r0>>(31-laneid)) & 0x1) << bIdx;
            }
        }
    }
}


// from compressed bit feature map (bit, M/32, N) --> (M, N) in uin32
__global__ void UnPackFcWeight128_OUTPUT(const uin32* __restrict__ A, uin32* B, 
                                  const int A_height, const int A_width, const int bitwidth)
{
    GET_LANEID;
    GET_WARPID;

    const int gdx = STEP128(A_height);
    const int gdy = STEP8(A_width);

    const int lx = (warpid & 0x3); // warp x_index vertical
    const int ly = (warpid >> 2);  // warp y_index horizontal

    const int offset_input = STEP128(A_height)*PAD8(A_width)*128/32;   // offset of input.

    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {
        const int bx = bid % gdx;
        const int by = bid / gdx;

        for (int bIdx = 0; bIdx < bitwidth; bIdx++){
            unsigned r0 = A[bIdx*offset_input + (by*8+ly)*gdx*4 + bx*4 + lx];
            // unsigned r0 = A[(bx*8+lx)*gdy*4+by*4+ly];
            if ((bx*128+lx*32+laneid<A_height) && (by*8+ly<A_width)){
                // B[bIdx * offset + (bx*8+lx)*A_width+by*128+ly*32+laneid] = 2*(uin32)((r0>>(31-laneid)) & 0x1) - 1; 
                B[(bx*128+lx*32+laneid)*A_width + by*8 + ly] += (uin32)((r0>>(31-laneid)) & 0x1) << bIdx;
            }
        }
    }
}



// from compressed bit feature map (bit, M, N/32) --> (M, N) in uin32
__global__ void UnPackFcOutput128(const uin32* __restrict__ A, uin32* B, 
                                  const int A_height, const int A_width, const int bitwidth)
{
    GET_LANEID;
    GET_WARPID;

    const int gdx = STEP8(A_height);
    const int gdy = STEP128(A_width);
    const int lx = (warpid >> 2);
    const int ly = (warpid & 0x3);

    const int offset_input = PAD8(A_height)*STEP128(A_width)*128/32;   // offset of input.

    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {
        const int bx = bid / gdy;
        const int by = bid % gdy;

        for (int bIdx = 0; bIdx < bitwidth; bIdx++){
            unsigned r0 = A[bIdx*offset_input + (bx*8+lx)*gdy*4 + by*4 + ly];
            // bitIdx * offset_opt + (bx*8+lx)*gdy*4+by*4+ly
            // unsigned r0 = A[(bx*8+lx)*gdy*4+by*4+ly];

            // (by*128+ly*32+laneid<(p->input_width)) 
            //         &&   (bx*8+lx<(p->input_height))
            if ((bx*8+lx<A_height) && (by*128+ly*32+laneid<A_width)){
                // B[bIdx * offset + (bx*8+lx)*A_width+by*128+ly*32+laneid] = 2*(uin32)((r0>>(31-laneid)) & 0x1) - 1; 
                B[(bx*8+lx)*A_width+by*128+ly*32+laneid] += (uin32)((r0>>(31-laneid)) & 0x1) << bIdx;
                // printf("r0: %u\n", r0);
                // printf("B[bIdx * offset + (bx*8+lx)*A_width+by*128+ly*32+laneid]: %u\n", B[bIdx * offset + (bx*8+lx)*A_width+by*128+ly*32+laneid]);
            }
        }
    }
}
/////////////////////////////////////////////////////
// input_gpu (float 32-bit) -->  input_qnt_gpu (uint 32-bit)
/////////////////////////////////////////////////////
__global__ void Quantize_val(uin32* input_qnt_gpu, float* input_gpu, int num_elements, int bitwidth){
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    // for all available threads.
    for (int tid = start; tid < num_elements; tid += blockDim.x * gridDim.x) {
    // if (tid < num_elements){
        /*
        * Quant_val  - 0            2^{bitwidth}    
        *-------------------- = ------------------
        * Actual_val - min_val  max_val - min_val
        */
        float input_val = clip(input_gpu[tid], min_v, max_v);
        float qnt_float = (input_val - min_v) * (1 << bitwidth) * 1.0f / (max_v - min_v);
        input_qnt_gpu[tid]  =  __float2uint_rn(qnt_float);
    }
}  


/////////////////////////////////////////////////////
// input_qnt_gpu (uint 32-bit) (M x N) -->  output_gpu (uint 32-bit) (bitwidth x M x N)
/////////////////////////////////////////////////////
__global__ void Decompose_bit(uin32* output_uin32_gpu, uin32* input_qnt_gpu, int num_elements, int bitwidth){
    int start = blockIdx.x * blockDim.x + threadIdx.x;

    // for all available threads.
    for (int tid = start; tid < num_elements; tid += blockDim.x*gridDim.x) {
    // if (tid < num_elements){
        for (int bIdx = 0; bIdx < bitwidth; bIdx++){
            output_uin32_gpu[bIdx*num_elements + tid] =  (input_qnt_gpu[tid] >> bIdx) & 0x01;
        } 
    }
}  



/////////////////////////////////////////////////////
// Convert floating point input into 1-bit input layer
// the height and width will not change for input layer
/////////////////////////////////////////////////////
class In128LayerParam
{
    public:
        In128LayerParam(const char* _name, int _input_height, int _input_width, int _bit_width)
            :input_height(_input_height), output_height(_input_height),
            input_width(_input_width), output_width(_input_width),
            input(NULL), input_gpu(NULL), output(NULL), output_gpu(NULL), bitwidth(_bit_width)
        {
            strncpy(name, _name, 8);
        }
        //input utility
        int input_size() { return input_height*input_width; }
        int input_bytes() { return input_size()*sizeof(float);}
        int input_bit_size() { return input_height*input_width; }
        int input_bit_bytes() { return input_bit_size()*sizeof(float);}
        //output utility
        int output_size() { return  output_height*output_width;}
        int output_bytes() { return output_size()*sizeof(float);}
        //binarize on row
        int output_bit_size() { return bitwidth*PAD8(output_height)*STEP128(output_width);}
        // add the bitwidth for computation (BW x compressed_feature_map)
        int output_bit_bytes() { return output_bit_size()*sizeof(uin128);}

        In128LayerParam* initialize(float* input)
        {
            cudaGetDeviceProperties(&deviceProp, dev);
            
            CHECK_NULL_POINTER(input);
            this->input = input;
            SAFE_ALOC_GPU(input_gpu, input_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(input_gpu, input, input_bytes(), cudaMemcpyHostToDevice) );
            // print_image_10x10_float(input);

            // input_gpu (float 32-bit) -->  input_qnt_gpu (uint 32-bit)
            dataQuantization();
            SAFE_ALOC_HOST(input_qnt, input_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(input_qnt, input_qnt_gpu, input_bytes(), cudaMemcpyDeviceToHost) );
            // print_image_10x10_int(input_qnt);

            // input_qnt_gpu (uint 32-bit) -->  output_gpu (packed 1-bit as uint 32-bit)
            bitDecomposition();
            SAFE_ALOC_HOST(output_uin32, bitwidth*output_height*output_width*sizeof(uin32));
            CUDA_SAFE_CALL( cudaMemcpy(output_uin32, output_uin32_gpu, bitwidth*output_height*output_width*sizeof(uin32), cudaMemcpyDeviceToHost));
            
            // print_image_10x10_int_bit_decompose(output_uin32);
            SAFE_ALOC_GPU(output_gpu, output_bit_bytes());
            // printf("In-layer output_gpu: %d, %d, %d\n", bitwidth, PAD8(output_height), output_width);

            // printf("In-layer output_gpu_byptes: %d\n", output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );
            return this->ready();
        }

        In128LayerParam* ready()
        {
            CHECK_NULL_POINTER(input_gpu);
            CHECK_NULL_POINTER(output_gpu);
            SAFE_ALOC_GPU(gpu, sizeof(In128LayerParam));
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, sizeof(In128LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }

        void set_output_gpu(uin32* _output_gpu) 
        { 
            this->output_gpu = _output_gpu; 
        }

        uin32* get_input_qnt(){
            return this->input_qnt;
        }
        
        uin32* get_output_gpu()
        { 
            return this->output_gpu; 
        }

        uin32* download_output()
        {
            if (output == NULL) 
                SAFE_ALOC_HOST(output, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, output_bit_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }
        
        /*Added quantization for quantize the initial value to N-bit representation in INT32*/
        void dataQuantization()
        {
            SAFE_ALOC_GPU(input_qnt_gpu, input_bytes());

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Quantize_val, numThreads, 0);            
            // input_gpu (float 32-bit) -->  input_qnt_gpu (uint 32-bit)
            Quantize_val<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(input_qnt_gpu, input_gpu, output_height*output_width, bitwidth);

            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());
        }

        /*Added quantization for decompose the INT32 matrix to N-bit x M x N bit matrix*/
        void bitDecomposition()
        {
            SAFE_ALOC_GPU(output_uin32_gpu, bitwidth*output_height*output_width*sizeof(uin32));
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Decompose_bit, numThreads, 0);
            Decompose_bit<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(output_uin32_gpu, input_qnt_gpu, output_height*output_width, bitwidth);
        }


        uin32* download_full_output()
        {
            const int size = output_bytes();

            uin32* full_output = NULL;
            SAFE_ALOC_HOST(full_output, size);

            uin32* full_output_gpu = NULL;
            SAFE_ALOC_GPU(full_output_gpu, size);
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, size) );

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128, 
                    numThreads, 0);
            UnPackFcOutput128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    output_gpu, full_output_gpu, output_height, output_width, bitwidth);

            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, size, cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaFree(full_output_gpu) );
            return full_output;
        }

        // print the image with float-point value.
        void print_image_10x10_float(float* image){
            printf("\n------print_image_10x10_float-----------\n");
            const int show_height = 28;
            const int show_width = show_height;
            for (int i = 0; i < show_height; i++){
                for (int j = 0; j < show_width; j++){
                    printf("%f ", image[i * 28 + j]);
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        // print the image with their int value.
        void print_image_10x10_int(uin32* image){
            printf("\n------print_image_10x10_int-----------\n");
            const int show_height = 10;
            const int show_width = show_height;
            for (int i = 0; i < show_height; i++){
                for (int j = 0; j < show_width; j++){
                    printf("%d ", image[i * 28 + j]); // * show the first image.
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        // print the image with their decomposed bit in 32-bit.
        void print_image_10x10_int_bit_decompose(uin32* image){
            const int show_height = 28;
            const int show_width = show_height;
            
            printf("\n------print_image_28x28_int_bit_decompose-----------\n");
            for (int i = 0; i < show_height; i++){
                for (int j = 0; j < show_width; j++){
                    for (int b = bitwidth - 1; b >= 0; b--)
                        printf("%d", image[b * output_width * output_height + i * 28 + j]); // * show the first image.
                    printf(" ");
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        void release() 
        {
            SAFE_FREE_GPU(input_gpu);
            SAFE_FREE_GPU(output_gpu);
            SAFE_FREE_GPU(gpu);
            SAFE_FREE_HOST(output);
        }

        ~In128LayerParam() { release(); }
    
    public:
        float* input;
        float* input_gpu;           // float input (M x N)

        uin32* input_qnt;
        uin32* input_qnt_gpu;       // quantized uint32 middle representation. (M x N)

        uin32*  output;
        uin32*  output_uin32;
        uin32*  output_uin32_gpu;     // uint32 matrix representation. (bitwidth x M x N)
        uin32*  output_gpu;          //  packed uint32 1-bit matrix representation. (bitwidth x M x N/32)
        
 
        int input_width;
        int input_height;
        int output_width;
        int output_height;
        int bitwidth;

        int numThreads = 1024;
        int numBlocksPerSm;
        cudaDeviceProp deviceProp;

        In128LayerParam* gpu;
        char name[8];
};

class Fc128LayerParam
{
    public:
        Fc128LayerParam(const char* name, int _input_height, int _input_width, 
                int _weight_width, int act_bit=2, int w_bit=2) : 
            weight_height(_input_width), weight_width(_weight_width), 
            input_height(_input_height), input_width(_input_width),
            output_height(_input_height), output_width(_weight_width),
            bn_width(_weight_width), weight(NULL), weight_gpu(NULL),
            bn(NULL), bn_gpu(NULL), output(NULL), output_gpu(NULL),
            input(NULL), input_gpu(NULL), gpu(NULL), act_bit(act_bit), w_bit(w_bit)
        {
            strncpy(this->name, name, 8);
        }
        //row major -- input
        int input_size() { return input_height*input_width;}
        int input_bytes() { return input_size()*sizeof(uin32);}
        int input_bit_size() { return PAD8(input_height)*STEP128(input_width);}
        int input_bit_bytes() { return input_bit_size()*sizeof(uin128);}

        //colum major -- weight
        int weight_size() { return weight_height*weight_width;}
        int weight_bytes() { return weight_size()*sizeof(float);}
        int weight_bit_size() { return w_bit*STEP128(weight_height)*PAD128(weight_width);}
        int weight_bit_bytes() { return weight_bit_size()*sizeof(uin128);}
        // int weight_bit_size() { return w_bit*(weight_height)*PAD8(weight_width);}
        // int weight_bit_bytes() { return weight_bit_size()*sizeof(uin128);}

        //row-major -- output
        int output_size() { return output_height*output_width;}
        int output_bytes() { return output_size()*sizeof(float);}
        // **using low-bit output
        // int output_bit_size() { return act_bit*PAD8(output_height)*STEP128(output_width);}
        // int output_bit_bytes() { return output_bit_size() * sizeof(uin128);}
        // **using int output
        int output_bit_size() { return act_bit*output_height*w_bit*output_width;}
        int output_bit_bytes() { return output_bit_size()*sizeof(float);}

        //batch-norm
        int bn_size() { return bn_width;}
        int bn_bytes() { return bn_size()*sizeof(float);}

        Fc128LayerParam* ready()
        {
            CHECK_NULL_POINTER(input_gpu);
            CHECK_NULL_POINTER(output_gpu);
            SAFE_ALOC_GPU(gpu, sizeof(Fc128LayerParam));
            // points to a GPU object
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, sizeof(Fc128LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }

        void set_input_gpu(uin32* _input_gpu)
        {
            this->input_gpu = _input_gpu;
        }

        Fc128LayerParam* initialize(FILE* config_file, uin32* prev_layer_gpu)
        {
            //Initialize weight[CPU] -- float32
            SAFE_ALOC_HOST(weight, weight_bytes());
            // launch_array(config_file, this->weight, weight_size());

            // Arbitarized weight_gpu [GPU]
            SAFE_ALOC_GPU(weight_gpu, weight_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->weight_gpu, 0, weight_bit_bytes()) );

            // printf("FC-layer weight_bit: %d, %d, %d\n", w_bit, weight_height, PAD128(weight_width));

            // Initialize weight_float [GPU] -- float32
            SAFE_ALOC_GPU(weight_float, weight_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(weight_float, weight, weight_bytes(), cudaMemcpyHostToDevice) );

            cudaGetDeviceProperties(&deviceProp, dev);

            // weight_float (float 32-bit) -->  quantized weight_qnt_gpu (uint 32-bit)
            weightQuantization();

            // check weight_uin32[CPU]  <--- weight_qnt_gpu[GPU]
            SAFE_ALOC_HOST(weight_uin32, weight_size()*sizeof(uin32));
            CUDA_SAFE_CALL( cudaMemcpy(weight_uin32, weight_qnt_gpu, weight_size()*sizeof(uin32), cudaMemcpyDeviceToHost));
            
            // printf("weight_height: %d, weight_width: %d\n", weight_height, weight_width);
            // printf("\n\nFC weight (float)");
            // print_image_10x10_float(weight);

            // printf("\n\nFC weight (quantied uint 32)");
            // print_image_10x10_int(weight_uin32);
            // exit(0);

            // weight_qnt_gpu (uint32) -->  weight_uin32_dec_gpu (packed 1-bit as uint 32-bit)
            weightBitDecomposition();

            SAFE_ALOC_HOST(weight_uin32_dec, w_bit*weight_size()*sizeof(uin32));
            CUDA_SAFE_CALL( cudaMemcpy(weight_uin32_dec, weight_uin32_dec_gpu, w_bit*weight_size()*sizeof(uin32), cudaMemcpyDeviceToHost));
            
            // printf("\n\nFC weight (bit decomposed uint 32)");
            // print_image_10x10_int_bit_decompose(weight_uin32_dec);

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, PackFcWeight128, 
                    numThreads, 0);
            PackFcWeight128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    weight_uin32_dec_gpu, weight_gpu, weight_height, weight_width, w_bit);

            // uin32* weight_cpu=NULL;
            // SAFE_ALOC_HOST(weight_cpu, weight_bit_bytes());
            // CUDA_SAFE_CALL( cudaMemcpy(weight_cpu, weight_gpu, weight_bit_bytes(), cudaMemcpyDeviceToHost) );
            // for (int i=0;i<weight_bit_size()*4; i++)
            //     printf("%u ", weight_cpu[i]);
            // exit(0);
            CUDA_CHECK_KERNEL();
            SAFE_FREE_GPU(weight_float);
            
            //Process bn
            SAFE_ALOC_HOST(bn, bn_bytes());
            // launch_array(config_file, this->bn, bn_size());
            SAFE_ALOC_GPU(bn_gpu, bn_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(bn_gpu, bn, bn_bytes(), cudaMemcpyHostToDevice) );

            // printf("FC-layer output_gpu: %d, %d, %d\n", act_bit, output_height, output_width);
            // printf("FC-layer output_gpu_byptes: %d\n", output_bit_bytes());
            //Allocate output gpu
            SAFE_ALOC_GPU(output_gpu, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );

            set_input_gpu(prev_layer_gpu);
            return this->ready();
        }

        /* quantization for quantize the initial value to N-bit representation in INT32*/
        void weightQuantization()
        {
            SAFE_ALOC_GPU(weight_qnt_gpu, weight_bytes());

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Quantize_val,  numThreads, 0);
            // input_gpu (float 32-bit) -->  input_qnt_gpu (uint 32-bit)
            Quantize_val<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(weight_qnt_gpu, weight_float, weight_size(), w_bit);  
            // printf("after quantize_val\n");
            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());
        }

        /*Added quantization for decompose the INT32 matrix to N-bit x M x N bit matrix*/
        void weightBitDecomposition()
        {
            SAFE_ALOC_GPU(weight_uin32_dec_gpu, w_bit * weight_size() * sizeof(uin32));

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Decompose_bit, numThreads, 0);

            Decompose_bit<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(weight_uin32_dec_gpu,  weight_qnt_gpu, weight_size(), w_bit);

            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());
        }

        // print the image with float-point value.
        void print_image_10x10_float(float* image){
            printf("\n------print_image_10x10_float-----------\n");
            const int show_height = 10;
            const int show_width = show_height;
            for (int i = 0; i < show_height; i++){
                for (int j = 0; j < show_width; j++){
                    printf("%f ", image[i * weight_height + j]);
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        // print the image with their decomposed bit in 32-bit.
        void print_weight_int_bit_decompose(uin32* image){
            const int show_height = weight_height;
            const int show_width = 1;
            // column store
            printf("\n------print_weight_int_bit_decompose-----------\n");
            for (int i = 0; i < show_width; i++){
                for (int j = 0; j < show_height; j++){
                    for (int b = w_bit - 1; b >= 0; b--)
                        // printf("%d", image[b * weight_size() + i * weight_height + j]);
                        printf("%u", image[b * weight_size() + i * show_height + j]);
                    printf(" ");
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        // print the image with their int value.
        void print_image_uint32(uin32* image, const int img_width){
            // printf("\n------print_image_%dx%d_int-----------\n", width, width);
            const int print_range = 8;
            for (int i = 0; i < print_range; i++){
                for (int j = 0; j < print_range; j++){
                    printf("%d ", image[i*img_width + j]);
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        uin32* get_output_gpu()
        {
            return this->output_gpu;
        }

        uin32* get_weight_uin32(){
            return this->weight_uin32;
        }

        uin32* download_output()
        {
            if (output == NULL) SAFE_ALOC_HOST(output, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, 
                        output_bit_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }

        uin32* download_full_weight()
        {
            const int size = weight_bytes();
            uin32* full_weight = NULL;
            SAFE_ALOC_HOST(full_weight, size);

            uin32* full_weight_gpu = NULL;
            SAFE_ALOC_GPU(full_weight_gpu, size);
            CUDA_SAFE_CALL( cudaMemset(full_weight_gpu, 0, size) ); 

#ifdef NEWFMT
            // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128FMT, 
            //         numThreads, 0);
            // UnPackFcOutput128FMT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
            //         output_gpu, full_output_gpu, output_height, output_width, 3);
#else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcWeight128, 
                    numThreads, 0);
            // * unpack weight for hidden layer
            UnPackFcWeight128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    weight_gpu, full_weight_gpu, weight_height, weight_width, w_bit);
#endif
            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());

            CUDA_SAFE_CALL( cudaMemcpy(full_weight, full_weight_gpu, size, cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaFree(full_weight_gpu) );

            return full_weight;
        }

        uin32* download_full_output()
        {
            const int size = output_size()*sizeof(uin32);
            
            // output (CPU) uint32
            uin32* full_output = NULL;
            SAFE_ALOC_HOST(full_output, size);

            // output (GPU) uint32
            uin32* full_output_gpu = NULL;
            SAFE_ALOC_GPU(full_output_gpu, size);
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, size) );
            
#ifdef NEWFMT
            // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128FMT, 
            //         numThreads, 0);
            // UnPackFcOutput128FMT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
            //         output_gpu, full_output_gpu, output_height, output_width, 3);
#else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128, 
                    numThreads, 0);
            UnPackFcOutput128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    output_gpu, full_output_gpu, output_height, output_width, act_bit);
#endif
            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, size, cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaFree(full_output_gpu) );

            return full_output;
        }

        void release()
        {
            SAFE_FREE_HOST(weight);
            SAFE_FREE_HOST(bn);
            SAFE_FREE_HOST(output);
            SAFE_FREE_GPU(weight_gpu);
            SAFE_FREE_GPU(bn_gpu);
            SAFE_FREE_GPU(output_gpu);
            SAFE_FREE_GPU(gpu);
        }
        ~Fc128LayerParam() { release(); }

    public:
    
        //Input
        uin32* input;
        uin32* input_gpu;
        int input_width;
        int input_height;
        
        //Weight
        float* weight;
        float* weight_float = NULL;
        uin32* weight_gpu;
        int weight_width;
        int weight_height;
        
        //Output
        uin32* output;
        uin32* output_gpu;
        int output_width;
        int output_height;

        // support for arbitary precision.
        int w_bit, act_bit;

        uin32* weight_uin32;
        uin32* weight_qnt_gpu;
        uin32* weight_uin32_dec;
        uin32* weight_uin32_dec_gpu;

        int numThreads = 1024;
        int numBlocksPerSm;
        cudaDeviceProp deviceProp;

        //Batch normalization
        float* bn;
        float* bn_gpu;
        int bn_width;

        //GPU shadow
        Fc128LayerParam* gpu;
        char name[8];
};


class Out128LayerParam
{
    public:
        Out128LayerParam(const char* name, int _input_height, 
                int _input_width, int _weight_width, int act_bit, int w_bit) :
            input_height(_input_height), input_width(_input_width),
            weight_height(_input_width), weight_width(_weight_width),
            output_height(_input_height), output_width(_weight_width),
            input(NULL), input_gpu(NULL), output(NULL), output_gpu(NULL),
            weight(NULL), weight_gpu(NULL), act_bit(act_bit), w_bit(w_bit)
        {
            strncpy(this->name, name, 8);
        }
        // row major
        int input_size() { return input_height*input_width;}
        int input_bytes() { return input_size()*sizeof(uin32);}
        int input_bit_size() { return act_bit*PAD8(input_height)*STEP128(input_width);}
        int input_bit_bytes() { return input_bit_size()*sizeof(uin128);}
        
        // colum major
        int weight_size() { return weight_height*weight_width;}
        int weight_bytes() { return weight_size()*sizeof(float);}
        int weight_bit_size() { return w_bit*STEP128(weight_height)*PAD8(weight_width);}
        int weight_bit_bytes() { return weight_bit_size()*sizeof(uin128);}

        // row major
        int output_size() { return output_height*output_width;}
        int output_bytes() { return output_size()*sizeof(float);}
        int output_bit_size() { return act_bit*output_height*output_width;}
        int output_bit_bytes() { return output_bit_size()*sizeof(float);}

        int bn_size() { return output_width;}
        int bn_bytes() { return output_width*sizeof(float); }
 
        Out128LayerParam* ready()
        {
            CHECK_NULL_POINTER(input_gpu);
            CHECK_NULL_POINTER(output_gpu);
            SAFE_ALOC_GPU(gpu, sizeof(Out128LayerParam));
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, sizeof(Out128LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }

        void set_input_gpu(uin32* input_gpu)
        {
            this->input_gpu = input_gpu;
        }

        uin32* get_output_gpu()
        {
            return this->output_gpu;
        }

        uin32* get_weight_uin32()
        {
            return this->weight_uin32;
        }

        Out128LayerParam* initialize(FILE* config_file, uin32* prev_layer_gpu)
        {

            // CPU weight -- float32
            SAFE_ALOC_HOST(weight, weight_bytes());
            // launch_array(config_file, this->weight, weight_size());
            // GPU weight -- arbitarized bit
            SAFE_ALOC_GPU(weight_gpu, weight_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->weight_gpu, 0, weight_bit_bytes()) );
            // GPU weight -- float32
            SAFE_ALOC_GPU(weight_float, weight_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(weight_float, weight, weight_bytes(), cudaMemcpyHostToDevice) );

            cudaGetDeviceProperties(&deviceProp, dev);

            // weight_float (float 32-bit) -->  quantized weight_qnt_gpu (uint 32-bit)
            weightQuantization();

            SAFE_ALOC_HOST(weight_uin32, weight_size()*sizeof(uin32));
            CUDA_SAFE_CALL( cudaMemcpy(weight_uin32, weight_qnt_gpu, weight_size()*sizeof(uin32), cudaMemcpyDeviceToHost));
            // printf("weight_height: %d, weight_width: %d\n", weight_height, weight_width);
            // print_image_10x10_int(weight_uin32);
            
            // weight_qnt_gpu (uint 32-bit) -->  weight_uin32_dec_gpu (packed 1-bit as uint 32-bit)
            weightBitDecomposition();

            SAFE_ALOC_HOST(weight_uin32_dec, w_bit*weight_size()*sizeof(uin32));
            CUDA_SAFE_CALL( cudaMemcpy(weight_uin32_dec, weight_uin32_dec_gpu, w_bit*weight_size()*sizeof(uin32), cudaMemcpyDeviceToHost));
            // printf("\n\nFC_out weight (bit decomposed uint 32)");
            // print_weight_int_bit_decompose(weight_uin32_dec);
#ifdef NEWFMT
            // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, PackFcWeight128FMT, 
            //         numThreads, 0);
            // PackFcWeight128FMT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
            //         weight_float, weight_gpu, weight_height, weight_width);
#else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, PackFcWeight128, 
                    numThreads, 0);
            PackFcWeight128_OUTPUT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    weight_uin32_dec_gpu, weight_gpu, weight_height, weight_width, w_bit);
#endif
            CUDA_CHECK_KERNEL();

            // printf("weight_bit size: %d ", weight_bit_size());
            // uin32* weight_cpu=NULL;
            // SAFE_ALOC_HOST(weight_cpu, weight_bit_bytes());
            // CUDA_SAFE_CALL( cudaMemcpy(weight_cpu, weight_gpu, weight_bit_bytes(), cudaMemcpyDeviceToHost) );
            // for (int i=0;i < weight_bit_size()*4; i++)
            //     printf("%u\n", weight_cpu[i]);
            // exit(0);

            //BN
            SAFE_ALOC_HOST(bn_scale, bn_bytes());
            // launch_array(config_file, this->bn_scale, bn_size());
            SAFE_ALOC_GPU(bn_scale_gpu, bn_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(bn_scale_gpu, bn_scale, bn_bytes(), cudaMemcpyHostToDevice) );

            SAFE_ALOC_HOST(bn_bias, bn_bytes());
            // launch_array(config_file, this->bn_bias, bn_size());
            SAFE_ALOC_GPU(bn_bias_gpu, bn_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(bn_bias_gpu, bn_bias, bn_bytes(), cudaMemcpyHostToDevice) );

            SAFE_ALOC_GPU(output_gpu, output_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bytes()) );
            set_input_gpu(prev_layer_gpu);

            return this->ready();
        }

         /* quantization for quantize the initial value to N-bit representation in INT32*/
        void weightQuantization()
        {
            SAFE_ALOC_GPU(weight_qnt_gpu, weight_bytes());
            
            // printf("out_weight_height: %d, out_weight_width: %d\n", weight_height, weight_width);
            cudaGetDeviceProperties(&deviceProp, dev);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Quantize_val,  numThreads, 0);
            
            // weight_gpu (float 32-bit) -->  weigth_qnt_gpu (uint 32-bit)
            Quantize_val<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(weight_qnt_gpu, weight_float, weight_size(), w_bit);

            // printf("after quantize_val\n");
            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());
        }

        /*Added quantization for decompose the INT32 matrix to N-bit x M x N bit matrix*/
        void weightBitDecomposition()
        {
            SAFE_ALOC_GPU(weight_uin32_dec_gpu, w_bit*weight_size()*sizeof(uin32));

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Decompose_bit, numThreads, 0);
            Decompose_bit<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(weight_uin32_dec_gpu, weight_qnt_gpu, weight_size(), w_bit);

            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());
        }

        // print the image with float-point value.
        void print_image_10x10_float(float* image){
            printf("\n------print_image_10x10_float-----------\n");
            const int show_height = 10;
            const int show_width = show_height;
            for (int i = 0; i < show_height; i++){
                for (int j = 0; j < show_width; j++){
                    printf("%f ", image[i * weight_height + j]);
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

       // print the image with float-point value.
        void print_image_10x10_float(float* image, const int width){
            printf("\n------print_image_10x10_float-----------\n");
            const int show_height = width;
            const int show_width = width;
            for (int i = 0; i < show_height; i++){
                for (int j = 0; j < show_width; j++){
                    printf("%f ", image[i * width + j]);
                }
                printf("\n");
            }
            printf("=========================================\n");
        }


        // print the image with their int value.
        void print_image_10x10_int(uin32* image){
            printf("\n------print_image_10x10_int-----------\n");
            const int show_height = 32;
            const int show_width = show_height;
            for (int i = 0; i < show_height; i++){
                for (int j = 0; j < show_width; j++){
                    printf("%d ", image[i * weight_height + j]);
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        // print the image with their int value.
        void print_output_int32(uin32* image){
            printf("\n------print_output_int32-----------\n");
            const int show_height = output_height;
            const int show_width = output_width;

            printf("show_height, %d, show_width: %d\n", show_height, show_width);
            for (int i = 0; i < show_height; i++){
                printf("[%d] ", i);
                for (int j = 0; j < show_width; j++){
                    printf("%u ", image[i * show_width + j]);
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        // print the image with their decomposed bit in 32-bit.
        void print_weight_int_bit_decompose(uin32* image){
            const int show_height = weight_height;
            const int show_width = 1;
            
            printf("\n------print_weight_int_bit_decompose-----------\n");
            for (int i = 0; i < show_width; i++){
                for (int j = 0; j < show_height; j++){
                    for (int b = w_bit - 1; b >= 0; b--)
                        printf("%d", image[b * weight_size() + i * weight_height + j]);
                    printf(" ");
                }
                printf("\n");
            }
            printf("=========================================\n");
        }
        
        // * print the input image with their decomposed bit-by-bit
        void print_input_int_bit_decompose(uin32* image){
            const int show_height = input_height;
            const int show_width = input_width;

            printf("show_height: %d, show_width: %d", show_height, show_width);
            
            printf("\n------print_input_int_bit_decompose-----------\n");
            for (int i = 0; i < show_height; i++){
                for (int j = 0; j < show_width; j++){
                    for (int b = act_bit - 1; b >= 0; b--) {
                        // printf("\n%d, %d, %d\n", i, j, b);
                        printf("%d", image[b * input_size() + i * show_width + j]);
                    }
                    printf(" ");
                }
                printf("\n");
            }
            printf("=========================================\n");
        }

        // convert a bit-by-bit matrix to uint32 matrix.
        uin32* bit2uint32(uin32* bit_input){

            uin32* uint32input =  NULL;
            SAFE_ALOC_HOST(uint32input, input_size() * sizeof(uin32));
            
            // row-major store
            printf("\n------Output FC Layer bit2uint32-----------\n");
            for (int i = 0; i < input_height; i++){
                for (int j = 0; j < input_width; j++){
                    uin32 tmp = 0;
                    for (int b = act_bit - 1; b >= 0; b--)
                        tmp += (bit_input[b * input_size() + i * input_width + j] << b);
                    uint32input[i * input_width + j] = tmp;
                }
            }
            return uint32input;
        }


        // print the image with their int value.
        void print_image_uint32(uin32* image, const int img_width){
            // printf("\n------print_image_%dx%d_int-----------\n", width, width);
            const int print_range = 8;
            for (int i = 0; i < print_range; i++){
                for (int j = 0; j < print_range; j++){
                    printf("%d ", image[i*img_width + j]);
                }
                printf("\n");
            }
            printf("=========================================\n");
        }


        uin32* download_full_weight()
        {
            const int size = weight_size() * sizeof(float);
            uin32* full_weight = NULL;
            SAFE_ALOC_HOST(full_weight, size);

            uin32* full_weight_gpu = NULL;
            SAFE_ALOC_GPU(full_weight_gpu, size);
            
            CUDA_SAFE_CALL( cudaMemset(full_weight_gpu, 0, size) );
            cudaGetDeviceProperties(&deviceProp, dev);
            
#ifdef NEWFMT
            // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128FMT, 
            //         numThreads, 0);
            // UnPackFcOutput128FMT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
            //         output_gpu, full_output_gpu, output_height, output_width, 3);
#else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128, numThreads, 0);
            UnPackFcWeight128_OUTPUT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    weight_gpu, full_weight_gpu, weight_height, weight_width, w_bit);
#endif
            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());

            CUDA_SAFE_CALL( cudaMemcpy(full_weight, full_weight_gpu, size, cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaFree(full_weight_gpu) );

            return full_weight;
        }

        // for validation input at output layer
        uin32* download_full_input()
        {
            const int size = input_bytes();
            uin32* full_input = NULL;
            SAFE_ALOC_HOST(full_input, size);

            uin32* full_input_gpu = NULL;
            SAFE_ALOC_GPU(full_input_gpu, size);
            CUDA_SAFE_CALL( cudaMemset(full_input_gpu, 0, size) );
            
#ifdef NEWFMT
            // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128FMT, 
            //         numThreads, 0);
            // UnPackFcOutput128FMT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
            //         output_gpu, full_output_gpu, output_height, output_width, 3);
#else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128, numThreads, 0);
            UnPackFcOutput128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    input_gpu, full_input_gpu, input_height, input_width, act_bit);
#endif
            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());

            CUDA_SAFE_CALL( cudaMemcpy(full_input, full_input_gpu, size, cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaFree(full_input_gpu) );
            
            return full_input;
        }
        
        //* validate output in int32 format.
        uin32* download_output()
        {
            SAFE_ALOC_HOST(output, output_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, output_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }

        void release()
        {
            SAFE_FREE_HOST(weight);
            SAFE_FREE_HOST(output);
            SAFE_FREE_HOST(bn_scale);
            SAFE_FREE_HOST(bn_bias);

            SAFE_FREE_GPU(weight_gpu);
            SAFE_FREE_GPU(output_gpu);
            SAFE_FREE_GPU(gpu);
            SAFE_FREE_GPU(bn_scale_gpu);
            SAFE_FREE_GPU(bn_bias_gpu);
        }
        ~Out128LayerParam() { release(); }
    public:
        //Input
        uin32* input;
        uin32* input_gpu;
        int input_width;
        int input_height;
        
        //Weight
        float* weight;
        float* weight_float=NULL;
        uin32* weight_gpu;
        int weight_width;
        int weight_height;

        //Output
        uin32* output;
        uin32* output_gpu;
        int output_height;
        int output_width;

        // support for arbitary precision.
        int w_bit, act_bit;

        uin32* weight_uin32;
        uin32* weight_qnt_gpu;
        uin32* weight_uin32_dec;
        uin32* weight_uin32_dec_gpu;

        int numThreads = 1024;
        int numBlocksPerSm;
        cudaDeviceProp deviceProp;

        //Batch normalization
        bool has_bn;
        float* bn_scale;
        float* bn_scale_gpu;
        float* bn_bias;
        float* bn_bias_gpu;

        //GPU shadow
        Out128LayerParam* gpu;
        char name[8];
};



//================================ Convolution ====================================

__global__ void PackFiltersByInChannels128(const uin32* __restrict__ filter, 
        unsigned* filter_binarized, const int input_channels, const int output_channels, 
        const int filter_width, const int filter_height, const int bitwidth) 
{
    GET_LANEID;
    GET_WARPID;
    const int bx = blockIdx.x;//iter over (filter_width*filter_height)
    const int by = blockIdx.y;//iter over output_channels
    const int ins = 4*STEP128(input_channels);//condense C:in_channel into 32bit-unsigned

    const int in_offset = filter_height * filter_width * input_channels * output_channels;
    const int out_offset = STEP32(in_offset);

    for (int c=0; c<ins; c++) //iter over C:input_channels
    {   
        for (int bIdx = 0; bIdx < bitwidth; bIdx++){
            // From shape[filter_height, filter_width, input_channels, output_channels] 
            uin32 f0 = ((c*32+laneid)<input_channels)? filter[bIdx*in_offset + bx*input_channels*output_channels 
                + (c*32+laneid)*output_channels + by]: -1;
            unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>0));
            // if (bIdx == 1){
            //     printf("bidx = 0, r0 = %u\n", r0);
            // }
            if (laneid == 0) //avoid warp conflict
                // To shape[filter_height, filter_width, output_channels, input_channels/32]
                // [K, K, O, C]
                // filter_binarized[bIdx*out_offset + bx*PAD32(output_channels)*ins + by*ins + c] = r0;

                //* new_CONV CONV kernel layout: bit x [O, K, K, C].
                filter_binarized[bIdx*out_offset + by*filter_height*filter_width*ins + bx*ins + c] = r0;
        }
    }
}

__global__ void UnpackFiltersByInChannels128(const uin32* __restrict__ filter_binarized, 
        unsigned* filter, const int input_channels, const int output_channels, 
        const int filter_width, const int filter_height, const int bitwidth) 
{
    GET_LANEID;
    GET_WARPID;

    const int bx = blockIdx.x;//iter over (filter_width*filter_height)
    const int by = blockIdx.y;//iter over output_channels
    const int ins = 4*STEP128(input_channels);//condense C:in_channel into 32bit-unsigned

    const int out_offset= filter_height * filter_width * input_channels * output_channels;
    const int in_offset = STEP32(out_offset);

    for (int c=0; c<ins; c++) //iter over C:input_channels
    {   
        for (int bIdx = 0; bIdx < bitwidth; bIdx++){
            // // From shape[filter_height, filter_width, input_channels, output_channels] 
            // uin32 f0 = ((c*32+laneid)<input_channels)? filter[bIdx*in_offset+bx*input_channels*output_channels 
            //     + (c*32+laneid)*output_channels + by]: 0;
            // unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>0));
            // if (laneid == 0) //avoid warp conflict
            //     // To shape[filter_height, filter_width, output_channels, input_channels/32]
            //     filter_binarized[bIdx*out_offset + bx*PAD32(output_channels)*ins + by*ins + c] = r0;

            if ((c*32 + laneid) < input_channels){
                // [K, K, O, C]
                uin32 r0 = filter_binarized[bIdx*in_offset + bx*PAD32(output_channels)*ins + by*ins + c];
                filter[bx*input_channels*output_channels + (c*32+laneid)*output_channels + by] += ((r0>>(31-laneid)) & 0x1) << bIdx;
                // filter[bIdx*out_offset + bx*input_channels*output_channels + (c*32+laneid)*output_channels + by] = (r0>>(31-laneid)) & 0x1;
            }
        }
    }
}


/*
__global__ void PackFiltersByOutChannels32(const float* __restrict__ filter, 
        unsigned* filter_binarized, const int in_channels, const int out_channels, 
        const int filter_width, const int filter_height) 
{
    unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));
    const int bx = blockIdx.x;//iter over (filter_width*filter_height)
    const int by = blockIdx.y;//iter over input_channels
    const int ots = STEP32(out_channels);//condense K:output_channel into 32bit-unsigned

    for (int k=0; k<ots*4; k++) //iter over K:output_channels
    {
        // From shape[filter_height, filter_width, in_channels, out_channels] 
        float f0 = ((k*32+laneid)<out_channels)? filter[bx*in_channels*out_channels 
            + by*out_channels + k*32 + laneid]:0;
        unsigned r0 = __brev(__ballot(f0>=0));
        // To shape[filter_height, filter_width, in_channels, out_channels/32]
        filter_binarized[bx*ots*in_channels+ by*ots + k] = r0;
    }
}  */


// compress the filter of the convolution weights.
// <<< dim3(filter_height*filter_width, input_channels), 32>>>
__global__ void PackFiltersByOutChannels32(const uin32* __restrict__ filter, 
        unsigned* filter_binarized, const int in_channels, const int out_channels, 
        const int filter_width, const int filter_height, const int w_bit) 
{
    GET_LANEID;
    GET_WARPID;
    const int bx = blockIdx.x; //iter over (filter_width*filter_height)
    const int by = blockIdx.y; //iter over input_channels
    const int ots = STEP32(out_channels);//condense K:output_channel into 32bit-unsigned

    const int in_offset = filter_width * filter_height * in_channels * out_channels;
    const int out_offset = STEP32(in_offset);

    for (int k=0; k<ots; k++) //iter over K:output_channels
    {
        for (int bIdx = 0; bIdx < w_bit; bIdx++){
            // From shape[filter_height, filter_width, in_channels, out_channels] 
            float f0 = ((k*32+laneid)<out_channels)? filter[bIdx*in_offset + bx*in_channels*out_channels 
                + by*out_channels + k*32 + laneid]:-1.0f;
            unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0));
            // To shape [filter_height, filter_width, in_channels, out_channels/32]
            // filter_binarized[bIdx*out_offset + bx*ots*in_channels + by*ots + k] = r0;

            //* for new CONV kernel --> bitwidth x [O/32, K, K, C]
            filter_binarized[bIdx*out_offset + k*filter_width*filter_height*in_channels + bx*in_channels + by] = r0;
        }

    }
}

// compress the filter of the convolution weights.
// <<< dim3(filter_height*filter_width, input_channels), 32>>>
__global__ void UnpackFiltersByOutChannels32(const uin32* __restrict__ filter_binarized, 
        unsigned* filter, const int in_channels, const int out_channels, 
        const int filter_width, const int filter_height, const int w_bit) 
{
    GET_LANEID;
    GET_WARPID;
    const int bx = blockIdx.x; //iter over (filter_width*filter_height)
    const int by = blockIdx.y; //iter over input_channels

    const int ots = STEP32(out_channels);//condense K:output_channel into 32bit-unsigned
    
    // output is uin32 uncompressed 
    const int out_offset = in_channels * out_channels * filter_width * filter_height;
    // input is 1-bit compressed as 1-bit
    const int in_offset = STEP32(out_offset);

    for (int k=0; k<ots; k++) //iter over K:output_channels
    {
        for (int bIdx = 0; bIdx < w_bit; bIdx++){
            // From shape[filter_height, filter_width, in_channels, out_channels] 
            // float f0 = ((k*32+laneid)<out_channels)? filter[bIdx * in_offset + bx*in_channels*out_channels 
                // + by*out_channels + k*32 + laneid]:-1.0f;
            if ((k*32+laneid) < out_channels){
                
                uin32 r0 = filter_binarized[bIdx*in_offset + bx*ots*in_channels + by*ots + k];
                // filter[bIdx*out_offset + bx*in_channels*out_channels + by*out_channels + k*32 + laneid] = (r0>>(31-laneid)) & 0x1;
                filter[bx*in_channels*out_channels + by*out_channels + k*32 + laneid] += ((r0>>(31-laneid)) & 0x1) << bIdx;
            }
        }
    }


}

// recover the output/input of a convolution layer.
// input_binarized (bitwidth x M x N/32) --> M x N
__global__ void UnpackConvOutput32(const unsigned* __restrict__ input_binarized, 
        uin32* input, const int input_height, const int input_width,
        const int input_channels, const int batch, const int bitwidth) 
{
    GET_LANEID;
    GET_WARPID;

    const int bx = blockIdx.x;//input_width
    const int by = blockIdx.y;//input_height
    const int bz = blockIdx.z;//batch

    /*const int ins = STEP32(input_channels);//condense C:in_channel into 32bit-unsigned*/
    const int ins = STEP128(input_channels);//condense C:in_channel into 32bit-unsigned

    const int input_offset = input_height*input_width*PAD8(batch)*STEP32(input_channels);

    /*const int otb = STEP8(batch);*/
    for (int c=0; c<ins*4; c++) //iter over C:in_channels
    {
        for (int bIdx = 0; bIdx < bitwidth; bIdx++){
            // From shape[input_height, input_width, batch, in_channels/32] 
            unsigned r0 = input_binarized[bIdx*input_offset + by*input_width*PAD8(batch)*ins*4 + bx*PAD8(batch)*ins*4 + bz*ins*4 + c];

            // To shape[batch, input_height, input_width, in_channels] --> [N, H, W, O]
            if (c*32+laneid<input_channels)
            {
                // // directly convert the 1-bit to 32-bit element.
                input[bz*input_height*input_width*input_channels + by*input_width*input_channels
                    + bx*input_channels + c*32 + laneid] += ((r0>>(31-laneid)) & 0x1) << bIdx;

                // int index = bz*input_height*input_width*input_channels 
                //         + by*input_width*input_channels 
                //         + bx*input_channels + c*32 + laneid;

                // // directly convert the 1-bit to 32-bit element.
                // input[] += ((r0>>(31-laneid)) & 0x1) << bIdx;


                // input[bz*input_height*input_width*input_channels + by*input_width*input_channels
                //     + bx*input_channels + c*32 + laneid] = 2*(float)((r0>>(31-laneid)) & 0x1)-1;
            }
        }
    }
}

//*----------------------------------------------------------
//*--------------Convoluation Param-------------------------//
//*----------------------------------------------------------
class InConv128LayerParam
{
    public:
        InConv128LayerParam(const char* name, int _input_height, int _input_width, 
                int _filter_height, int _filter_width, int _input_channels, 
                int _output_channels, int _batch, int act_bit=3, int w_bit=2, int _stride_height=1, 
                int _stride_width=1, bool _padding=true, int _pool_height=1, 
                int _pool_width=1, bool _save_residual=false) :
            input_height(_input_height), input_width(_input_width), filter_height(_filter_height),
            filter_width(_filter_width), input_channels(_input_channels),
            output_channels(_output_channels), batch(_batch), stride_height(_stride_height),
            stride_width(_stride_width), pool_height(_pool_height), pool_width(_pool_width),
            save_residual(_save_residual), padding(_padding), 
            bn(NULL), filter(NULL), output(NULL), output_gpu(NULL), input(NULL), 
            input_gpu(NULL), gpu(NULL), output_residual_gpu(NULL), act_bit(act_bit), w_bit(w_bit)

        {
            strncpy(this->name, name, 8);

            // * check whether to pad additional element. 
            this->pad_h = padding?((( (input_height+stride_height-(input_height%stride_height))
                            /stride_height-1)*stride_height+filter_height-input_height)/2):0;
            this->pad_w = padding?((( (input_width+stride_width-(input_width%stride_width))
                                /stride_width-1)*stride_width+filter_width-input_width)/2):0; 

            int buf_height = padding?(input_height+stride_height-1)/stride_height
                    :((input_height-filter_height)/stride_height+1);
            int buf_width = padding?(input_width+stride_width-1)/stride_width
                    :((input_width-filter_width)/stride_width+1);

            output_height = (buf_height+pool_height-1)/pool_height;//pooling height
            output_width = (buf_width+pool_width-1)/pool_width; //pooling width
        }
        InConv128LayerParam* ready()
        {
            CHECK_NULL_POINTER(input_gpu);
            CHECK_NULL_POINTER(output_gpu);
            if (save_residual) CHECK_NULL_POINTER(output_residual_gpu);
            SAFE_ALOC_GPU(this->gpu, sizeof(InConv128LayerParam));
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, 
                        sizeof(InConv128LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }
        // input feature map.
        int input_size() { return input_channels*input_height*input_width*batch;}
        int input_bytes() { return input_size()*sizeof(float);}
        int input_bit_size() { return act_bit*input_channels*input_height*input_width*batch;}
        int input_bit_bytes() {return input_bit_size()*sizeof(float);}

        //filter weight
        int filter_size() { return output_channels*input_channels*filter_height*filter_width;}
        int filter_bytes() { return filter_size()*sizeof(float);}
        int filter_bit_size() {return w_bit*PAD128(output_channels)*STEP128(input_channels)*filter_height*filter_width;}
        int filter_bit_bytes() { return filter_bit_size() * sizeof(uin128);}
        
        // output feature map
        int output_size() { return output_channels*output_height*output_width*batch;}
        int output_bytes() { return output_size()*sizeof(uin32);}
        int output_bit_size() { return act_bit*STEP128(output_channels)*output_height*output_width*PAD8(batch); }
        int output_bit_bytes() { return output_bit_size() * sizeof(uin128); }

        // batch normal layer
        int bn_size() { return output_channels;}
        int bn_bytes() { return bn_size()*sizeof(float);}
        
        // residual_layer
        int residual_size() { return PAD128(output_channels)*PAD8(batch)*output_height*output_width;}
        int residual_bytes() { return residual_size()*sizeof(int);}

        InConv128LayerParam* initialize(float* input, FILE* config_file)
        {
            //Process input
            CHECK_NULL_POINTER(input);
            this->input = input;
            SAFE_ALOC_GPU(input_gpu, input_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(input_gpu, input, input_bytes(), cudaMemcpyHostToDevice) );

            //Process weight (compressed to 1-bit)
            SAFE_ALOC_HOST(filter, filter_bytes());
            // launch_array(config_file, this->filter, filter_size());
            SAFE_ALOC_GPU(filter_gpu, filter_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(filter_gpu, 0, filter_bit_bytes()) );

            // non-quantizated filter weights.
            SAFE_ALOC_GPU(filter_float, filter_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(filter_float, filter, filter_bytes(), cudaMemcpyHostToDevice) );

            // show input
            // print_image_float(filter);

            // quantization on float --> uint32
            filterQuantization();
            // validate
            uin32* filter_qnt = NULL;
            SAFE_ALOC_HOST(filter_qnt, filter_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(filter_qnt, filter_qnt_gpu, filter_bytes(), cudaMemcpyDeviceToHost) );
            // print_image_int(filter_qnt);
            // <---end validation 

            // bit-decomposition on uint32.
            filterBitDecomposition();
            // validate
            uin32* filter_uin32_dec = NULL;
            SAFE_ALOC_HOST(filter_uin32_dec, w_bit * filter_size() * sizeof(uin32));
            CUDA_SAFE_CALL( cudaMemcpy(filter_uin32_dec, filter_uin32_dec_gpu, w_bit * filter_size() * sizeof(uin32), cudaMemcpyDeviceToHost));
            // print_image_bit_decompose(filter_uin32_dec, filter_size(), w_bit);
            // <---end validation 

            //Binarize Filter
            PackFiltersByOutChannels32<<<dim3(filter_height*filter_width, input_channels), 32>>>(
                            filter_uin32_dec_gpu, filter_gpu, input_channels, 
                            output_channels, filter_width, filter_height, w_bit);

            SAFE_FREE_GPU(filter_float);

            uin32* full_filter = download_full_filter();
            // validate
            // print_image_bit_decompose(full_filter, filter_size(), w_bit);
            // print_image_int(full_filter);
            // <---end validation 
            
            //Process bn in [float]
            SAFE_ALOC_HOST(bn, bn_bytes());
            // launch_array(config_file, bn, bn_size());
            SAFE_ALOC_GPU(bn_gpu, bn_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(this->bn_gpu, this->bn, bn_bytes(), cudaMemcpyHostToDevice) );

            //Allocate output in [arbitary-bit].
            SAFE_ALOC_GPU(output_gpu, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );
            
            //Allocate residual for saving
            if (save_residual)
            {
                SAFE_ALOC_GPU(output_residual_gpu, residual_bytes());
                CUDA_SAFE_CALL( cudaMemset(this->output_residual_gpu, 0, residual_bytes()) );
            }

            // printf("\n=================================\n");
            return this->ready();
        }

        uin32* get_output_gpu() { return this->output_gpu; }
        int* get_output_residual_gpu() { return this->output_residual_gpu; }

         /* quantization for quantize the initial value to N-bit representation in INT32*/
        void filterQuantization()
        {
            SAFE_ALOC_GPU(filter_qnt_gpu, filter_bytes());
            
            // printf("out_weight_height: %d, out_weight_width: %d\n", weight_height, weight_width);
            cudaGetDeviceProperties(&deviceProp, dev);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Quantize_val,  numThreads, 0);
            
            // input_gpu (float 32-bit) -->  input_qnt_gpu (uint 32-bit)
            Quantize_val<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(filter_qnt_gpu, filter_float, filter_size(), w_bit);

            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());
        }

        /*Added quantization for decompose the INT32 matrix to N-bit x M x N bit matrix*/
        void filterBitDecomposition()
        {
            SAFE_ALOC_GPU(filter_uin32_dec_gpu, w_bit * filter_size() * sizeof(uin32));

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Decompose_bit, numThreads, 0);

            Decompose_bit<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(filter_uin32_dec_gpu, filter_qnt_gpu, filter_size(), w_bit);

            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());
        }

        // print the image with float-point value.
        void print_image_float(float* image){
            printf("\n------print_image_float-----------\n");
            for (int i = 0; i < 10; i++){
                printf("%.3f ", image[i]);
            }
            printf("\n");
        }

        // print the image with float-point value.
        void print_image_int(uin32* image){
            printf("\n------print_image_int-----------\n");
            for (int i = 0; i < 10; i++){
                printf("%u ", image[i]);
            }
            printf("\n");
        }

        // print the image with their decomposed bit in 32-bit.
        void print_image_bit_decompose(uin32* image, int offset, int bitwidth){
            printf("\n------print_image_int_bit_decompose-----------\n");
            for (int i = 0; i < 10; i++){
                for (int b = bitwidth - 1; b >= 0; b--)
                    printf("%u", image[b*offset + i]); 
                printf(" ");
            }
            printf("\n");
        }

        // check input in float.
        float* download_full_input() { return this->input;}

        unsigned* download_output()
        {
            if (output == NULL) SAFE_ALOC_HOST(output, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, output_bit_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }

        uin32* download_full_filter()
        {
            int size = filter_size()*sizeof(uin32);

            uin32* full_filter = NULL;
            SAFE_ALOC_HOST(full_filter, size);

            uin32* full_filter_gpu = NULL;
            SAFE_ALOC_GPU(full_filter_gpu, size);
            CUDA_SAFE_CALL( cudaMemset(full_filter_gpu, 0, size) );

#ifdef NEWFMT
//             UnpackConvOutput32FMT<<<dim3(output_width,output_height,batch), 32>>>(output_gpu,
//                     full_output_gpu, output_height, output_width, output_channels, batch);
#else
            UnpackFiltersByOutChannels32<<<dim3(filter_height*filter_width, input_channels), 32>>>(filter_gpu,
                    full_filter_gpu, input_channels, output_channels, filter_width, filter_height, w_bit);
#endif

            CUDA_SAFE_CALL( cudaMemcpy(full_filter, full_filter_gpu, size, cudaMemcpyDeviceToHost) );
            SAFE_FREE_GPU(full_filter_gpu);
            
            return full_filter;
        }

        uin32* download_full_output()
        {
            uin32* full_output = NULL;
            SAFE_ALOC_HOST(full_output, output_bytes());

            uin32* full_output_gpu = NULL;
            SAFE_ALOC_GPU(full_output_gpu, output_bytes());
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, output_bytes()) );

#ifdef NEWFMT
            // UnpackConvOutput32FMT<<<dim3(output_width,output_height,batch), 32>>>(output_gpu,
            //         full_output_gpu, output_height, output_width, output_channels, batch);
#else
            UnpackConvOutput32<<<dim3(output_width,output_height,batch), 32>>>(output_gpu,
                    full_output_gpu, output_height, output_width, output_channels, batch, act_bit);
#endif
            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, output_bytes(), cudaMemcpyDeviceToHost) );
            SAFE_FREE_GPU(full_output_gpu);

            return full_output;
        }

        void release()
        {
            SAFE_FREE_HOST(filter);
            SAFE_FREE_HOST(bn);
            SAFE_FREE_HOST(output);
            SAFE_FREE_GPU(input_gpu);
            SAFE_FREE_GPU(output_gpu);
            SAFE_FREE_GPU(filter_gpu);
            SAFE_FREE_GPU(bn_gpu);
            SAFE_FREE_GPU(gpu);
            if (save_residual) 
                SAFE_FREE_GPU(output_residual_gpu);
        }
        ~InConv128LayerParam() { release(); }
    public:
        // arbitary support
        int act_bit, w_bit;

        // input.
        float* input;
        float* input_gpu;
        int input_width;
        int input_height;
        int input_channels;

        // filter weights
        float* filter;
        float* filter_float = NULL;
        uin32* filter_gpu;
        int filter_width;
        int filter_height;
        uin32* filter_qnt_gpu;          // quantized filter on GPU.
        uin32* filter_uin32_dec_gpu;    // bit decomposed filter on GPU.

        // output feature map.
        uin32* output;
        uin32* output_gpu;
        int output_width;
        int output_height;
        int output_channels;
        bool padding;

        float* bn;
        float* bn_gpu;

        int batch;
        int stride_height;
        int stride_width;
        int pad_h;
        int pad_w;
        int pool_width;
        int pool_height;
        bool save_residual;
        int* output_residual_gpu;

        int numThreads = 1024;
        int numBlocksPerSm;
        cudaDeviceProp deviceProp;


        InConv128LayerParam* gpu;
        char name[8];
};

class Conv128LayerParam
{
    public:
        Conv128LayerParam(const char* name, int _input_height, int _input_width, 
                int _filter_height, int _filter_width, int _input_channels, 
                int _output_channels, int _batch, int act_bit=2, int w_bit=2,
                int _stride_height=1, 
                int _stride_width=1, bool _padding=true, int _pool_height=1, 
                int _pool_width=1, bool _ahead_fc=false, bool _save_residual=false,
                bool _inject_residual=false, int _residual_channels=0,
                bool _residual_pool=false) :

            input_height(_input_height), input_width(_input_width), 
            filter_height(_filter_height), filter_width(_filter_width),
            input_channels(_input_channels), output_channels(_output_channels),
            batch(_batch), stride_height(_stride_height), stride_width(_stride_width),
            pool_height(_pool_height), pool_width(_pool_width), ahead_fc(_ahead_fc),
            save_residual(_save_residual), inject_residual(_inject_residual),
            residual_channels(_residual_channels), padding(_padding), 
            residual_pool(_residual_pool),
            bn(NULL), bn_gpu(NULL), filter(NULL), filter_gpu(NULL), output(NULL),
            output_gpu(NULL), input(NULL), input_gpu(NULL), gpu(NULL), 
            output_residual_gpu(NULL), input_residual_gpu(NULL),
            act_bit(act_bit), w_bit(w_bit)
                
        {
            strncpy(this->name, name, 8);

            // compute padding if needed.
            this->pad_h = padding?((( (input_height+stride_height-(input_height%stride_height))
                            /stride_height-1)*stride_height+filter_height-input_height)/2):0;
            this->pad_w = padding?((( (input_width+stride_width-(input_width%stride_width))
                                /stride_width-1)*stride_width+filter_width-input_width)/2):0; 

            int buf_height = padding?(input_height+stride_height-1)/stride_height
                :((input_height-filter_height)/stride_height+1);
            output_height = (buf_height+pool_height-1)/pool_height;//pooling height
            int buf_width = padding?(input_width+stride_width-1)/stride_width
                :((input_width-filter_width)/stride_width+1);

            output_width = (buf_width+pool_width-1)/pool_width; //pooling width
        }
        // input related.
        int input_size() { return input_channels*input_height*input_width*batch;}
        int input_bytes() { return input_size()*sizeof(uin32);}
        int input_bit_size() { return act_bit*STEP128(input_channels)*input_height*input_width*PAD8(batch); }
        int input_bit_bytes() { return input_bit_size()*sizeof(uin128);}

        // filter related.
        int filter_size() { return output_channels*input_channels*filter_height*filter_width;}
        int filter_bytes() { return filter_size()*sizeof(float);}
        int filter_bit_size() { return w_bit*PAD32(output_channels)*STEP128(input_channels)*filter_height*filter_width; }
        int filter_bit_bytes() { return filter_bit_size()*sizeof(uin128);}

        // output related.
        int output_size() { return output_channels*output_height*output_width*batch;}
        int output_bytes() { return output_size()*sizeof(uin32);}
        int output_bit_size() { return act_bit*STEP128(output_channels)*output_height*output_width*PAD8(batch); }
        int output_bit_bytes() { return output_bit_size()*sizeof(uin128); }

        // bn and 
        int bn_size() { return output_channels;}
        int bn_bytes() { return bn_size()*sizeof(float);}

        // residual connections.
        int residual_size() { return act_bit*PAD128(output_channels)*PAD8(batch)*output_height*output_width;}
        int residual_bytes() { return residual_size()*sizeof(int);}

        Conv128LayerParam* ready()
        {
            CHECK_NULL_POINTER(input_gpu);
            CHECK_NULL_POINTER(output_gpu);
            if (save_residual) CHECK_NULL_POINTER(output_residual_gpu);
            if (inject_residual) CHECK_NULL_POINTER(input_residual_gpu);
            SAFE_ALOC_GPU(this->gpu, sizeof(Conv128LayerParam));
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, sizeof(Conv128LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }

        void set_input_gpu(uin32* input_gpu) { this->input_gpu = input_gpu; }
        void set_input_residual_gpu(int* input_residual_gpu) { this->input_residual_gpu = input_residual_gpu; }

        Conv128LayerParam* initialize(FILE* config_file, uin32* prev_layer_gpu,
                int* input_residual_gpu = NULL)
        {
            //Process weight
            SAFE_ALOC_HOST(filter, filter_bytes());
            // launch_array(config_file, filter, filter_size());
            SAFE_ALOC_GPU(filter_gpu, filter_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(filter_gpu, 0, filter_bit_bytes()) );

            SAFE_ALOC_GPU(filter_float, filter_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(filter_float, filter, filter_bytes(), cudaMemcpyHostToDevice) );

            // show filter in float
            // print_image_float(filter);

            // quantization on float --> uint32
            filterQuantization();
            // validate
            SAFE_ALOC_HOST(filter_qnt, filter_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(filter_qnt, filter_qnt_gpu, filter_bytes(), cudaMemcpyDeviceToHost) );
            // print_image_int(filter_qnt);
            // <---end validation 

            // bit-decomposition on uint32.
            filterBitDecomposition();

            // validate
            uin32* filter_uin32_dec = NULL;
            SAFE_ALOC_HOST(filter_uin32_dec, w_bit * filter_size() * sizeof(uin32));
            CUDA_SAFE_CALL( cudaMemcpy(filter_uin32_dec, filter_uin32_dec_gpu, w_bit * filter_size() * sizeof(uin32), cudaMemcpyDeviceToHost));
            // print_image_bit_decompose(filter_uin32_dec, filter_size(), w_bit);
            // <---end validation 
            
#ifdef NEWFMT
            // PackFiltersByInChannels128FMT<<<dim3(filter_height*filter_width, output_channels),
            //     32>>>(filter_float, filter_gpu, input_channels, output_channels, 
            //         filter_width, filter_height);
#else
            PackFiltersByInChannels128<<<dim3(filter_height*filter_width, output_channels), 32>>>(filter_uin32_dec_gpu, filter_gpu, input_channels, output_channels, filter_width, filter_height, w_bit);
#endif
            SAFE_FREE_GPU(filter_float);

            uin32* full_filter = download_full_filter();
            // validate
            // print_image_int(full_filter);            
            // print_image_bit_decompose(full_filter, filter_size(), w_bit);

            //Process bn
            SAFE_ALOC_HOST(bn, bn_bytes());
            // launch_array(config_file, bn, bn_size());
            SAFE_ALOC_GPU(bn_gpu, bn_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(bn_gpu, bn, bn_bytes(), cudaMemcpyHostToDevice) );

            //Allocate output gpu
            SAFE_ALOC_GPU(output_gpu, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );
            set_input_gpu(prev_layer_gpu);
            
            //Allocate residual for saving
            if (save_residual)
            {
                SAFE_ALOC_GPU(output_residual_gpu, residual_bytes());
                CUDA_SAFE_CALL( cudaMemset(output_residual_gpu, 0, residual_bytes()) );
            }
            //inject residual
            if (inject_residual) set_input_residual_gpu(input_residual_gpu);

            // printf("\n=================================\n");
            return this->ready();
        }
        uin32* get_output_gpu() { return this->output_gpu; }
        int* get_output_residual_gpu() { return this->output_residual_gpu; }

         /* quantization for quantize the initial value to N-bit representation in INT32*/
        void filterQuantization()
        {
            SAFE_ALOC_GPU(filter_qnt_gpu, filter_bytes());
            
            // printf("out_weight_height: %d, out_weight_width: %d\n", weight_height, weight_width);
            cudaGetDeviceProperties(&deviceProp, dev);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Quantize_val,  numThreads, 0);
            
            // input_gpu (float 32-bit) -->  input_qnt_gpu (uint 32-bit)
            Quantize_val<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(filter_qnt_gpu, filter_float, filter_size(), w_bit);

            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());
        }

        /*Added quantization for decompose the INT32 matrix to N-bit x M x N bit matrix*/
        void filterBitDecomposition()
        {
            SAFE_ALOC_GPU(filter_uin32_dec_gpu, w_bit * filter_size() * sizeof(uin32));

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Decompose_bit, numThreads, 0);

            Decompose_bit<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(filter_uin32_dec_gpu, filter_qnt_gpu, filter_size(), w_bit);

            CUDA_SAFE_CALL( cudaPeekAtLastError());
            CUDA_SAFE_CALL( cudaDeviceSynchronize());
        }

        // print the image with float-point value.
        void print_image_float(float* image){
            printf("\n------print_image_float-----------\n");
            for (int i = 0; i < 10; i++){
                printf("%.3f ", image[i]);
            }
            printf("\n");
        }

        // print the image with float-point value.
        void print_image_int(uin32* image){
            printf("\n------print_image_int-----------\n");
            for (int i = 0; i < 10; i++){
                printf("%u ", image[i]);
            }
            printf("\n");
        }

        // print the image with their decomposed bit in 32-bit.
        void print_image_bit_decompose(uin32* image, int offset, int bitwidth){
            printf("\n------print_image_int_bit_decompose-----------\n");
            for (int i = 0; i < 10; i++){
                for (int b = bitwidth - 1; b >= 0; b--)
                    printf("%u", image[b*offset + i]); 
                printf(" ");
            }
            printf("\n");
        }
        
        uin32* download_full_input()
        {
            int size = input_size()*sizeof(uin32);
            uin32* full_input = NULL;
            SAFE_ALOC_HOST(full_input, size);

            uin32* full_input_gpu = NULL;
            SAFE_ALOC_GPU(full_input_gpu, size);
            CUDA_SAFE_CALL( cudaMemset(full_input_gpu, 0, size) );
#ifdef NEWFMT
//             UnpackConvOutput32FMT<<<dim3(output_width,output_height,batch), 32>>>(output_gpu,
//                     full_output_gpu, output_height, output_width, output_channels, batch);
#else
            UnpackConvOutput32<<<dim3(input_height*input_width,batch), 32>>>(input_gpu,
                    full_input_gpu, input_height, input_width, input_channels, batch, act_bit);
#endif

            CUDA_SAFE_CALL( cudaMemcpy(full_input, full_input_gpu, size, cudaMemcpyDeviceToHost) );
            SAFE_FREE_GPU(full_input_gpu);

            return full_input;
        }

        unsigned* download_output()
        {
            if (output == NULL) SAFE_ALOC_HOST(output, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, output_bit_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }

        uin32* download_full_filter()
        {
            int size = filter_size()*sizeof(uin32);
            uin32* full_filter = NULL;
            SAFE_ALOC_HOST(full_filter, size);

            uin32* full_filter_gpu = NULL;
            SAFE_ALOC_GPU(full_filter_gpu, size);
            CUDA_SAFE_CALL( cudaMemset(full_filter_gpu, 0, size) );
#ifdef NEWFMT
//             UnpackConvOutput32FMT<<<dim3(output_width,output_height,batch), 32>>>(output_gpu,
//                     full_output_gpu, output_height, output_width, output_channels, batch);
#else
            UnpackFiltersByInChannels128<<<dim3(filter_height*filter_width, output_channels), 32>>>(filter_gpu,
                    full_filter_gpu, input_channels, output_channels, filter_width, filter_height, w_bit);
#endif
            CUDA_SAFE_CALL( cudaMemcpy(full_filter, full_filter_gpu, size, cudaMemcpyDeviceToHost) );
            SAFE_FREE_GPU(full_filter_gpu);
            return full_filter;
        }


        uin32* download_full_output()
        {
            uin32* full_output = NULL;
            SAFE_ALOC_HOST(full_output, output_bytes());

            uin32* full_output_gpu = NULL;
            SAFE_ALOC_GPU(full_output_gpu, output_bytes());
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, output_bytes()));

            UnpackConvOutput32<<<dim3(output_width,output_height,batch), 32>>>(output_gpu,
                    full_output_gpu, output_height, output_width, output_channels, batch, act_bit);

            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, output_bytes(), cudaMemcpyDeviceToHost) );
            
            SAFE_FREE_GPU(full_output_gpu);
            return full_output;
        }

        void release()
        {
            SAFE_FREE_HOST(filter);
            SAFE_FREE_HOST(bn);
            SAFE_FREE_HOST(output);
            SAFE_FREE_GPU(output_gpu);
            SAFE_FREE_GPU(filter_gpu);
            SAFE_FREE_GPU(bn_gpu);
            SAFE_FREE_GPU(gpu);
            if (save_residual) SAFE_FREE_GPU(output_residual_gpu);
        }
        ~Conv128LayerParam() { release(); }

    public:
        // arbitary support
        int act_bit, w_bit;

        //Input
        uin32* input;
        uin32* input_gpu;
        int input_width;
        int input_height;
        int input_channels;

        //Weight
        float* filter;
        float* filter_float = NULL; // filter_float on GPU
        uin32* filter_gpu;
        int filter_width;
        int filter_height;
        uin32* filter_qnt_gpu;          // quantized filter on GPU.
        uin32* filter_uin32_dec_gpu;    // bit decomposed filter on GPU.
        uin32* filter_qnt = NULL;       // quantized filter on CPU
        
        //Output
        uin32* output;
        uin32* output_gpu;
        int output_width;
        int output_height;
        int output_channels;

        int numThreads = 1024;
        int numBlocksPerSm;
        cudaDeviceProp deviceProp;

        // bn layer
        float* bn;
        float* bn_gpu;
        int batch;
        int stride_height;
        int stride_width;
        bool padding;
        int pad_h;
        int pad_w;
        int pool_width;
        int pool_height;
        bool ahead_fc;
        bool save_residual;
        int* output_residual_gpu;
        bool inject_residual;
        int* input_residual_gpu; 
        int residual_channels;
        bool residual_pool;
        Conv128LayerParam* gpu;
        char name[8];
};








#endif
