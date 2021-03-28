// ---------------------------------------------------------------------------
// File: cifar10_vgg.cu
// VGG-Net BNN inference source file for CIFAR10.
// ---------------------------------------------------------------------------
// See our arXiv paper for detail: https://arxiv.org/abs/2006.16578
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/TCBNN
// PNNL-IPID: 31925-E, ECCN: EAR99, IR: PNNL-SA-152850
// BSD Lincese.
// Richland, 99352, WA, USA. June-30-2020.
// ---------------------------------------------------------------------------

#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
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
// #include "new_kernel_conv.cu"

// #define VALIDATION
#define NUM_PROF 100

using namespace cooperative_groups;
using namespace std;

// #define w1_a2
#define w2_a2
// #define w1_a4

#ifdef w1_a2
const unsigned w_bit = 1;
const unsigned act_bit = 2;
// #include "51_APCONV_w1a2_t64.cu"        // 0.074 ms
// #include "52_APCONV_w1a2_t128.cu"       // 0.057 ms

#include "31_APMM_w1a2_t128.cu" 
#endif

#ifdef w2_a2
const unsigned w_bit = 2;
const unsigned act_bit = 2;
// #include "48_APCONV_w2a2_t64.cu"         // 0.050 ms
// #include "49_APCONV_w2a2_t128.cu"        // 0.063 ms
#include "50_APCONV_w2a2_t128.cu"        // 0.063 ms

#include "13_APMM_w2a2_t64.cu"  //0.013 ms
// #include "29_APMM_w2a2_t128.cu"    //0.018 ms
// #include "30_APMM_w2a2_t128.cu" //0.008 ms
#endif

#ifdef w1_a4
const unsigned w_bit = 1;
const unsigned act_bit = 4;
// #include "53_APCONV_w1a4_t64.cu"            // 0.041 ms
#include "54_APCONV_w1a4_t128.cu"        // 0.098 ms only for batch-size >= 8

// #include "32_APMM_w1a4_t128.cu"           
#include "33_APMM_w1a4_t64.cu"        
#endif

#ifdef w1_a8
const unsigned w_bit = 1;
const unsigned act_bit = 8;
// #include "55_APCONV_w1a8_t64.cu"        //  ms
// #include "56_APCONV_w1a8_t128.cu"        // ms
#endif

__global__ void vggnet128(
        InConv128LayerParam* bconv1, 
        Conv128LayerParam* bconv2, 
        Conv128LayerParam* bconv3,
        Conv128LayerParam* bconv4, 
        Conv128LayerParam* bconv5, 
        Conv128LayerParam* bconv6,
        Fc128LayerParam* bfc1, 
        Fc128LayerParam* bfc2, 
        Out128LayerParam* bout)
{
    grid_group grid = this_grid();

    #define newKernel
    #ifdef newKernel
    //SET_KERNEL_TIMER;
    //========= Conv1 ============
    InConv128Layer(bconv1);
    grid.sync();

    //TICK_KERNEL_TIMER(bconv1);
    //========= Conv2 ============
    Conv128Layer_new(bconv2);
    grid.sync();

    //TICK_KERNEL_TIMER(bconv2);
    //========= Conv3 ============
    Conv128Layer_new(bconv3);
    grid.sync();

    //TICK_KERNEL_TIMER(bconv3);
    //========= Conv4 ============
    // Conv128Layer(bconv4);
    Conv128Layer_new(bconv4);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv4);
    //========= Conv5 ============
    Conv128Layer_new(bconv5);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv5);

    //========= Conv6 ============
    Conv128Layer_new(bconv6);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv6);

    //========= Fc1 ============
    // Fc128Layer(bfc1);
    Fc128Layer_new(bfc1);
    grid.sync();
    //TICK_KERNEL_TIMER(bfc1);
    //========= Fc2 ============
    // Fc128Layer(bfc2);
    Fc128Layer_new(bfc2);
    grid.sync();
    //TICK_KERNEL_TIMER(bfc2);
    ////========== Output ===========
    Out128Layer_new(bout);
    // Out128Layer(bout);
    //TICK_KERNEL_TIMER(bout);

    #else

    //SET_KERNEL_TIMER;
    //========= Conv1 ============
    InConv128Layer(bconv1);
    grid.sync();

    //TICK_KERNEL_TIMER(bconv1);
    //========= Conv2 ============
    Conv128Layer(bconv2);
    grid.sync();

    //TICK_KERNEL_TIMER(bconv2);
    //========= Conv3 ============
    Conv128Layer(bconv3);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv3);

    //========= Conv4 ============
    Conv128Layer(bconv4);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv4);

    //========= Conv5 ============
    Conv128Layer(bconv5);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv5);

    //========= Conv6 ============
    Conv128Layer(bconv6);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv6);
    
    //========= Fc1 ============
    Fc128Layer(bfc1);
    grid.sync();
    //TICK_KERNEL_TIMER(bfc1);
    
    //========= Fc2 ============
    Fc128Layer(bfc2);
    grid.sync();
    
    //TICK_KERNEL_TIMER(bfc2);
    //========== Output ===========
    Out128Layer(bout);
    //TICK_KERNEL_TIMER(bout);
    #endif
}



int main()
{
    int dev = 0;
    cudaSetDevice(dev);

    const unsigned batch = 8;
    const unsigned output_size = 10;
    const unsigned image_height = 32;
    const unsigned image_width = 32;
    const unsigned image_channel = 3;
    const unsigned filter_height = 3;
    const unsigned filter_width = 3;
    const unsigned n_hidden = 1024;

    //=============== Get Input and Label =================
    float* images = (float*)malloc(batch*image_height*image_width*image_channel*sizeof(float));
    unsigned* image_labels = (unsigned*)malloc(batch*sizeof(unsigned));
    string cifar10_dir = "/home/yuke/.data/cifar10c/test_batch.bin";
    read_CIFAR10_normalized(cifar10_dir, images, image_labels, batch);

    //================ Get Weight =================
    //FILE* config_file = fopen("./cifar10.config","r");
    FILE* config_file = fopen("./vgg_cifar10.csv","r");

    //================ Set Network =================
    //Bconv1 Layer. (N, 3, H, W) --> (N, 128, H_1, W_1)
    printf("Layer,N,H,W,C,K,R,S\n");

    InConv128LayerParam* bconv1 = new InConv128LayerParam("Conv1", image_height, image_width, filter_height, filter_width, 3, 128, batch, act_bit, w_bit); 
    InConv128LayerParam* bconv1_gpu = bconv1->initialize(images, config_file);
    printf("Conv1,%d,%d,%d,%d,%d,%d,%d\n",  batch, image_height, image_width, 3, 128, filter_height, filter_width);

    //Bconv2 Layer
    Conv128LayerParam* bconv2 = new Conv128LayerParam("Conv2", bconv1->output_height, bconv1->output_width, filter_height, filter_width, 128, 128, batch, act_bit, w_bit, 1, 1, true, 2, 2, false); // temporarily change for pool->1 for debugging
    Conv128LayerParam* bconv2_gpu = bconv2->initialize(config_file, bconv1->get_output_gpu());
    printf("Conv2,%d,%d,%d,%d,%d,%d,%d\n",  batch, bconv1->output_height, bconv1->output_width, 128, 128, filter_height, filter_width);

    printf("act_bit: %d, w_bit: %d\n", act_bit, w_bit);
    //Bconv3 Layer
    Conv128LayerParam* bconv3 = new Conv128LayerParam("Conv3", bconv2->output_height, bconv2->output_width, filter_height, filter_width, 128, 256, batch, act_bit, w_bit);
    Conv128LayerParam* bconv3_gpu = bconv3->initialize(config_file, bconv2->get_output_gpu());
    printf("Conv3,%d,%d,%d,%d,%d,%d,%d\n",  batch, bconv2->output_height, bconv2->output_width, 128, 256, filter_height, filter_width);

    //Bconv4 Layer
    Conv128LayerParam* bconv4 = new Conv128LayerParam("Conv4", bconv3->output_height, bconv3->output_width, filter_height, filter_width, 256, 256, batch, act_bit, w_bit, 1, 1, true, 2, 2, false);
    Conv128LayerParam* bconv4_gpu = bconv4->initialize(config_file, bconv3->get_output_gpu());
    printf("Conv4,%d,%d,%d,%d,%d,%d,%d\n",  batch, bconv3->output_height, bconv3->output_width, 256, 256, filter_height, filter_width);

    //Bconv5 Layer
    Conv128LayerParam* bconv5 = new Conv128LayerParam("Conv5", bconv4->output_height, bconv4->output_width, filter_height, filter_width, 256, 512, batch, act_bit, w_bit);
    Conv128LayerParam* bconv5_gpu = bconv5->initialize(config_file, bconv4->get_output_gpu());
    printf("Conv5,%d,%d,%d,%d,%d,%d,%d\n",  batch, bconv4->output_height, bconv4->output_width, 256, 512, filter_height, filter_width);

    //Bconv6 Layer
    Conv128LayerParam* bconv6 = new Conv128LayerParam("Conv6", bconv5->output_height, bconv5->output_width, filter_height, filter_width, 512, 512, batch, act_bit, w_bit, 1, 1, true, 2, 2, true);
    Conv128LayerParam* bconv6_gpu = bconv6->initialize(config_file, bconv5->get_output_gpu());
    printf("Conv6,%d,%d,%d,%d,%d,%d,%d\n",  batch, bconv5->output_height, bconv5->output_width, 512, 512, filter_height, filter_width);

    //Fc1 Layer
    Fc128LayerParam* bfc1 = new Fc128LayerParam("Fc1", batch, (bconv6->output_height)*(bconv6->output_width)*512, n_hidden, act_bit, w_bit); 
    Fc128LayerParam* bfc1_gpu = bfc1->initialize(config_file, bconv6->get_output_gpu());
    printf("FC1,%d,%d,%d\n",  batch, (bconv6->output_height)*(bconv6->output_width)*512, n_hidden);

    //Fc2 Layer
    Fc128LayerParam* bfc2 = new Fc128LayerParam("Fc2", batch, n_hidden, n_hidden, act_bit, w_bit); 
    Fc128LayerParam* bfc2_gpu = bfc2->initialize(config_file, bfc1->get_output_gpu());
    printf("FC2,%d,%d,%d\n",  batch, n_hidden, n_hidden);

    //Out Layer
    Out128LayerParam* bout = new Out128LayerParam("Fout", batch, n_hidden, output_size, act_bit, w_bit);
    Out128LayerParam* bout_gpu = bout->initialize(config_file, bfc2->get_output_gpu());  
    printf("FCout,%d,%d,%d\n",  batch, n_hidden, output_size);


    //================ Setup Kernel =================
    int numThreads = 32;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    //int shared_memory = 512*sizeof(int)*32;
    int shared_memory =  100000; // 256*sizeof(int)*32;
    cudaFuncSetAttribute(vggnet128, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, vggnet128, numThreads, shared_memory);
    printf("\n========= numBlocksPerSm:%d ==========\n", numBlocksPerSm);

    void* args[] = {&bconv1_gpu, &bconv2_gpu, &bconv3_gpu, &bconv4_gpu, &bconv5_gpu, &bconv6_gpu,
        &bfc1_gpu, &bfc2_gpu, &bout_gpu};

    START_TIMER;
    for (int i=0; i<NUM_PROF; i++)
    cudaLaunchCooperativeKernel((void*)vggnet128, numBlocksPerSm*deviceProp.multiProcessorCount, 
            numThreads, args, shared_memory);

    STOP_TIMER;
    CUDA_CHECK_KERNEL();
    printf("Time: %.3f (ms)\n", milliseconds/NUM_PROF);


    #ifdef VALIDATION
    // 
    // <--- convlution layer-1
    // 
    // verify INPUT convolution layer input.
    float* conv1_input = bconv1->download_full_input();
    // print_first_item_input(conv1_input, bconv1);

    uin32* conv1_filter = bconv1->download_full_filter();
    // print_first_filter<InConv128LayerParam>(conv1_filter, bconv1);

    uin32* reference_conv1 = perform_dummy_convolution_inputLayer(conv1_input, conv1_filter, bconv1);
    uin32* conv1_output = bconv1->download_full_output();

    // print_first_item_output_uin32<InConv128LayerParam>(conv1_output, bconv1);
    // print_first_item_output_uin32<InConv128LayerParam>(reference_conv1, bconv1);

//     bool validatation_result = validate_difference(reference_conv1, conv1_output, bconv1->output_size());
//     if (validatation_result) 
//         printf("conv1 validation passed!\n");
//     else 
//         printf("conv1 validation failed\n");




    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // <--- convlution layer-2
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    uin32* conv2_filter = bconv2->download_full_filter();
    // print_first_filter<Conv128LayerParam>(conv2_filter, bconv2);

    // verify OUTPUT convolution layer output.
    uin32* conv2_output = bconv2->download_full_output();
    print_first_item_output_uin32<Conv128LayerParam>(conv2_output, bconv2);

    // conv1_output_CPU [N, H, W, O]
    uin32* reference_conv2 = perform_dummy_convolution_hiddenLayer(conv1_output, conv2_filter, bconv2);
    print_first_item_output_uin32<Conv128LayerParam>(reference_conv2, bconv2);

    // validate the correctness for CONV-2.
    validate_difference(reference_conv2, conv2_output, bconv2->output_size());
    
    #endif
/*
    float* out = bfc1->download_full_output();
    for (int i=65536; i<65536+256; i++)
    //for (int i=8192; i<8192+256; i++)
    {
        printf("%.f ", out[i]);
        if ((i+1)%16==0) printf("\n");
    }

    printf("\n===%f===\n", bout->bn_scale[0]);
*/


    //================ Output =================
//     uin32 *output = bout->download_output();
    //validate_prediction(output, image_labels, output_size, batch);


    //for (int i=0; i<256; i++)
    //{
    //printf("%f ",output[i]);
    //if ((i+1)%10==0) printf("\n");
    //}

    delete bconv1;
    delete bconv2;
    delete bconv3;
    delete bconv4;
    delete bconv5;
    delete bconv6;
    delete bfc1;
    delete bfc2;
    delete bout;

    return 0;
}

