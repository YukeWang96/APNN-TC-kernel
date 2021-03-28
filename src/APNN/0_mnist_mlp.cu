#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <cooperative_groups.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "utility.h"
#include "param.h"
#include "kernel.cuh"
#include "data.h"
#include "validation.h"

// import new kernel.
// #include "new_kernel_gemm.cu"
#include "input128Layer.cu"

#define w1_a2
// #define w2_a2
// #define w1_a4

#ifdef w1_a2
const unsigned w_bit = 1;
const unsigned act_bit = 2;
#include "31_APMM_w1a2_t128.cu" // 0.019 ms only work on batch = 128
#endif

#ifdef w2_a2
const unsigned w_bit = 2;
const unsigned act_bit = 2;
// #include "13_APMM_w2a2_t64.cu"  //0.013 ms
#include "29_APMM_w2a2_t128.cu"    //0.018 ms
// #include "30_APMM_w2a2_t128.cu" //0.008 ms
#endif

#ifdef w1_a4
const unsigned w_bit = 1;
const unsigned act_bit = 4;
#include "32_APMM_w1a4_t128.cu"            //0.021 ms
// #include "33_APMM_w1a4_t64.cu"         //0.013 ms
#endif

// #define VALIDATION  // whether to validate
#define NUM_PROF 1000   

using namespace cooperative_groups;
using namespace std;

#ifndef MAX
#define MAX(a,b) ((a)>(b)?(a):(b))
#endif

#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif


__global__ void mnist_mlp(In128LayerParam* bin, Fc128LayerParam* fc1, Fc128LayerParam* fc2, 
        Fc128LayerParam* fc3, Out128LayerParam* bout)
{
    grid_group grid = this_grid();

    #define newKernel
    #ifdef newKernel
        //========= Input ============
        In128Layer_new(bin);
        grid.sync();
        
        // ========== FC1 ============
        Fc128Layer_new(fc1);
        grid.sync();

        // ========== FC2 ============
        Fc128Layer_new(fc2);
        grid.sync();

        //========== FC3 ============
        Fc128Layer_new(fc3);
        grid.sync();

        // ========== Output ===========
        Out128Layer_new(bout);
        grid.sync();
    #else
        //========= Input ============
        In128Layer(bin);
        grid.sync();
        
        // ========== FC1 ============
        Fc128Layer(fc1);
        grid.sync();

        //========== FC2 ============
        Fc128Layer(fc2);
        grid.sync();

        //========== FC3 ============
        Fc128Layer(fc3);
        grid.sync();

        //========== Output ===========
        Out128Layer(bout);
        grid.sync();
    #endif
}


int main()
{
    //=============== Configuration =================
    int dev = 0;
    cudaSetDevice(dev);
    const unsigned batch = 128; // 1 --> misaligned address, 8 is OK.
    const unsigned output_size = 10;
    const unsigned n_hidden = 1024;
    const unsigned image_height = 32;
    const unsigned image_width = 32;
    const unsigned image_size = image_height*image_width;


    //=============== Get Input and Label =================
    string mnist_dir = "/home/yuke/.data/mnist/t10k-images-idx3-ubyte";
    float* images = NULL;
    SAFE_ALOC_HOST(images, image_height*image_width*batch*sizeof(float));
    string mnist_label = "/home/yuke/.data/mnist/t10k-labels-idx1-ubyte";
    unsigned* image_labels = NULL;
    SAFE_ALOC_HOST(image_labels, batch*sizeof(unsigned));
    read_MNIST_normalized(mnist_dir, mnist_label, images, image_labels, batch);

    //================ Get Weight =================
    FILE* config_file = fopen("./mlp_mnist.csv","r");

    //================ Set Network =================
    //Input Layer
    In128LayerParam* bin = new In128LayerParam("Fin", batch, image_size, act_bit);
    In128LayerParam* bin_gpu = bin->initialize(images);

    // Fc1 Layer
    Fc128LayerParam* bfc1 = new Fc128LayerParam("Fc1", batch, image_size, n_hidden, act_bit, w_bit); 
    Fc128LayerParam* bfc1_gpu = bfc1->initialize(config_file, bin->get_output_gpu());
    
    //Fc2 Layer
    Fc128LayerParam* bfc2 = new Fc128LayerParam("Fc2", batch, n_hidden, n_hidden, act_bit, w_bit); 
    Fc128LayerParam* bfc2_gpu = bfc2->initialize(config_file, bfc1->get_output_gpu());
    // Fc128LayerParam* bfc2_gpu = NULL;

    //Fc3 Layer
    Fc128LayerParam* bfc3 = new Fc128LayerParam("Fc3", batch, n_hidden, n_hidden, act_bit, w_bit); 
    Fc128LayerParam* bfc3_gpu = bfc3->initialize(config_file, bfc2->get_output_gpu());
    // Fc128LayerParam* bfc3_gpu = NULL;
    
    //Out Layer
    Out128LayerParam* bout = new Out128LayerParam("Fout", batch, n_hidden, output_size, act_bit, w_bit);
    Out128LayerParam* bout_gpu = bout->initialize(config_file, bfc3->get_output_gpu());
    // Out128LayerParam* bout_gpu = NULL;

    //================ Setup Kernel =================
    int numThreads = 128;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    // int shared_memory = 64*sizeof(int)*32; // <-- only 8KB not enough
    int shared_memory = 100000;   // maximum shared memory size for RTX3090 = 100 KB, first layer cost around 64KB.


    // SHMEM_SZ = MAX(sizeof(int4) * (BLOCK_COL_TILES * M) *
    // (CHUNK_K * (K/128) + SKEW) * 2, M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(int))


    cudaGetDeviceProperties(&deviceProp, dev);
    cudaFuncSetAttribute(mnist_mlp, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, mnist_mlp, numThreads, shared_memory);

    void* args[] = {&bin_gpu, &bfc1_gpu, &bfc2_gpu, &bfc3_gpu, &bout_gpu};

    START_TIMER;
    for (int i=0; i<NUM_PROF; i++)
    cudaLaunchCooperativeKernel((void*)mnist_mlp, numBlocksPerSm*deviceProp.multiProcessorCount, numThreads, args, shared_memory);
    STOP_TIMER;
    CUDA_CHECK_KERNEL();

    printf("Time: %.3f (ms)\n", milliseconds/NUM_PROF);

    //================ validation =================
    #ifdef VALIDATION
    // First FC layer
    uin32 *full_output_bin = bin->download_full_output();
    uin32 *full_weight_fc1 = bfc1->download_full_weight();
    uin32 *full_output_fc1 = bfc1->download_full_output();
    uin32 *full_output_fc1_ref = perform_dummy_FC_inputlayer<Fc128LayerParam>(full_output_bin, full_weight_fc1, bfc1);

    bfc1->print_image_uint32(full_output_fc1_ref, bfc1->output_width);
    bfc1->print_image_uint32(full_output_fc1, bfc1->output_width);

    bool val1 = validate_difference(full_output_fc1, full_output_fc1_ref, (bfc1->output_width)*(bfc1->output_height));
    printf("=============================\n");
    if (val1) printf("=> val1 [PASS!\n");
    else printf("val1 FAILED\n");
    printf("=============================\n");

    // -----------------
    // OUTPUT FC layer
    uin32* full_output_fc3 = bfc3->download_full_output();
    uin32* full_output_weight = bout->download_full_weight();
    // printf("=> %u\n", full_output_weight[0]);
    
    uin32* output = bout->download_output();
    uin32* output_ref = perform_dummy_FC_Layer<Out128LayerParam>(full_output_fc3, full_output_weight, bout);

    bout->print_image_uint32(output, bout->output_width);
    bout->print_image_uint32(output_ref, bout->output_width);

    bool val2 = validate_difference(output, output_ref, (bout->output_width)*(bout->output_height));
    printf("=============================\n");
    if (val2) printf("=> val2 [PASS!\n");
    else printf("val2 FAILED\n");
    printf("=============================\n");

    #endif // END: ifdef VALIDATION


    //================ Release =================
    delete bin;
    delete bfc1;
    delete bfc2;
    delete bfc3;
    delete bout;

    SAFE_FREE_HOST(image_labels);
    SAFE_FREE_HOST(images);

    printf("Finished running!\n");

    return 0;
}
