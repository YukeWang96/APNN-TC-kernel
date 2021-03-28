// ---------------------------------------------------------------------------
// File: alexnet.cu
// AlexNet BNN inference source file for ImageNet. 
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

using namespace cooperative_groups;
using namespace std;

#define NUM_PROF 100
#define w1_a2
// #define w2_a2
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

__global__ void alexnet128(
        InConv128LayerParam* bconv1, 
        Conv128LayerParam* bconv2, 
        Conv128LayerParam* bconv3,
        Conv128LayerParam* bconv4, 
        Conv128LayerParam* bconv5, 
        Fc128LayerParam* bfc1, 
        Fc128LayerParam* bfc2, 
        Out128LayerParam* bout)
{
    grid_group grid = this_grid();
    //========= Conv1 ============
    InConv128Layer(bconv1);
    grid.sync();
    //========= Conv2 ============
    Conv128Layer(bconv2);
    grid.sync();
    //========= Conv3 ============
    Conv128Layer(bconv3);
    grid.sync();
    //========= Conv4 ============
    Conv128Layer(bconv4);
    grid.sync();
    //========= Conv5 ============
    Conv128Layer(bconv5);
    grid.sync();
    //========= Fc1 ============
    Fc128Layer(bfc1);
    grid.sync();
    //========= Fc2 ============
    Fc128Layer(bfc2);
    grid.sync();
    //========== Output ===========
    Out128Layer(bout);
}

int main()
{
    int dev = 0;
    cudaSetDevice(dev);

    const unsigned batch = 128;
    const unsigned output_size = 1000;
    const unsigned image_height = 224;
    const unsigned image_width = 224;
    const unsigned image_channel = 3;
    const unsigned n_hidden = 4096;

    //=============== Get Input and Label =================
    float* images = (float*)malloc(batch*image_height*image_width*image_channel*sizeof(float));
    unsigned* image_labels = (unsigned*)malloc(batch*sizeof(unsigned));
//     read_ImageNet_normalized("./imagenet_files.txt", images, image_labels, batch);

    //================ Get Weight =================
    FILE* config_file = fopen("./alexnet_imagenet.csv","r");

    //================ Set Network =================
    //Bconv1 Layer
    InConv128LayerParam* bconv1 = new InConv128LayerParam("Conv1", image_height, image_width, 
            11, 11, 3, 64, batch, act_bit, w_bit, 4, 4, true, 2, 2); 
    InConv128LayerParam* bconv1_gpu = bconv1->initialize(images, config_file);

    //Bconv2 Layer
    Conv128LayerParam* bconv2 = new Conv128LayerParam("Conv2", bconv1->output_height, 
            bconv1->output_width, 5, 5, 64, 192, batch, act_bit, w_bit, 1, 1,
            true, 2, 2);
    Conv128LayerParam* bconv2_gpu = bconv2->initialize(config_file, bconv1->get_output_gpu());

    //Bconv3 Layer
    Conv128LayerParam* bconv3 = new Conv128LayerParam("Conv3", bconv2->output_height, 
            bconv2->output_width, 3, 3, 192, 384, batch, act_bit, w_bit);
    Conv128LayerParam* bconv3_gpu = bconv3->initialize(config_file, bconv2->get_output_gpu());

    //Bconv4 Layer
    Conv128LayerParam* bconv4 = new Conv128LayerParam("Conv4", bconv3->output_height, 
            bconv3->output_width, 3, 3, 384, 256, batch, act_bit, w_bit);
    Conv128LayerParam* bconv4_gpu = bconv4->initialize(config_file, bconv3->get_output_gpu());

    //Bconv5 Layer
    Conv128LayerParam* bconv5 = new Conv128LayerParam("Conv5", bconv4->output_height, 
            bconv4->output_width, 3, 3, 256, 256, batch, act_bit, w_bit, 1, 1, true, 2, 2, true);
    Conv128LayerParam* bconv5_gpu = bconv5->initialize(config_file, bconv4->get_output_gpu());

    //Fc1 Layer
    Fc128LayerParam* bfc1 = new Fc128LayerParam("Fc1", batch, (bconv5->output_height)
            *(bconv5->output_width)*256, n_hidden, act_bit, w_bit); 
    Fc128LayerParam* bfc1_gpu = bfc1->initialize(config_file, bconv5->get_output_gpu());
    
    //Fc2 Layer
    Fc128LayerParam* bfc2 = new Fc128LayerParam("Fc2", batch, n_hidden, n_hidden, act_bit, w_bit); 
    Fc128LayerParam* bfc2_gpu = bfc2->initialize(config_file, bfc1->get_output_gpu());
    //Out Layer
    Out128LayerParam* bout = new Out128LayerParam("Fout", batch, n_hidden, output_size, act_bit, w_bit);
    Out128LayerParam* bout_gpu = bout->initialize(config_file, bfc2->get_output_gpu());  

    //================ Setup Kernel =================
    int numThreads = 128;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    int shared_memory = 512*sizeof(int)*32;
    cudaFuncSetAttribute(alexnet128, cudaFuncAttributeMaxDynamicSharedMemorySize,shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, alexnet128, numThreads, shared_memory);

    void* args[] = {&bconv1_gpu, &bconv2_gpu, &bconv3_gpu, &bconv4_gpu, &bconv5_gpu, 
        &bfc1_gpu, &bfc2_gpu, &bout_gpu};


    START_TIMER;
    for (int i=0; i<NUM_PROF; i++)
    cudaLaunchCooperativeKernel((void*)alexnet128, numBlocksPerSm*deviceProp.multiProcessorCount, 
            numThreads, args, shared_memory);
    CUDA_CHECK_KERNEL();
    STOP_TIMER;
    printf("Time: %.3f (ms)\n", milliseconds/NUM_PROF);

    //================ Output =================
//     float* output = bout->download_output();
    //validate_prediction(output, image_labels, output_size, batch);

    /*
    float* out = bconv2->download_full_output();
    //for (int i=0; i<512; i++)
    for (int i=4096; i<4096+512; i++)
    {
        printf("%.f ", out[i]);
        if ((i+1)%32==0) printf("\n");
    }
    printf("\n===%f===\n", bout->bn_scale[0]);
    */

    delete bconv1;
    delete bconv2;
    delete bconv3;
    delete bconv4;
    delete bconv5;
    delete bfc1;
    delete bfc2;
    delete bout;

    return 0;
}
