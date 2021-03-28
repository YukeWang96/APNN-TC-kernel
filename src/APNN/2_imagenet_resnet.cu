// ---------------------------------------------------------------------------
// File: imagenet_resnet.cu
// ResNet-18 BNN inference source file for ImageNet. 
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

#define NUM_PROF 100

// #define w1_a2
#define w2_a2
// #define w1_a4

#ifdef w1_a2
const unsigned w_bit = 1;
const unsigned act_bit = 2;
#include "51_APCONV_w1a2_t64.cu"           // 0.152 ms
// #include "52_APCONV_w1a2_t128.cu"       // 

#include "31_APMM_w1a2_t128.cu" 
#endif

#ifdef w2_a2
const unsigned w_bit = 2;
const unsigned act_bit = 2;
#include "48_APCONV_w2a2_t64.cu"             // 0.209 ms
// #include "49_APCONV_w2a2_t128.cu"         //
// #include "50_APCONV_w2a2_t128.cu"        //

#include "13_APMM_w2a2_t64.cu"  //0.013 ms
// #include "29_APMM_w2a2_t128.cu"    //0.018 ms
// #include "30_APMM_w2a2_t128.cu" //0.008 ms
#endif

#ifdef w1_a4
const unsigned w_bit = 1;
const unsigned act_bit = 4;
#include "53_APCONV_w1a4_t64.cu"            // 
// #include "54_APCONV_w1a4_t128.cu"        // 

#include "33_APMM_w1a4_t64.cu"      
// #include "32_APMM_w1a4_t128.cu"           
#endif

#ifdef w1_a8
const unsigned w_bit = 1;
const unsigned act_bit = 8;
// #include "55_APCONV_w1a8_t64.cu"        //  ms
// #include "56_APCONV_w1a8_t128.cu"        // ms
#endif


using namespace cooperative_groups;
using namespace std;

__global__ void resnet128(
        InConv128LayerParam* bconv1, 
        Conv128LayerParam* l1b1c1, 
        Conv128LayerParam* l1b1c2,
        Conv128LayerParam* l1b2c1, 
        Conv128LayerParam* l1b2c2,
        Conv128LayerParam* l2b1c1, 
        Conv128LayerParam* l2b1c2,
        Conv128LayerParam* l2b2c1, 
        Conv128LayerParam* l2b2c2,
        Conv128LayerParam* l3b1c1, 
        Conv128LayerParam* l3b1c2,
        Conv128LayerParam* l3b2c1, 
        Conv128LayerParam* l3b2c2,
        Conv128LayerParam* l4b1c1, 
        Conv128LayerParam* l4b1c2,
        Conv128LayerParam* l4b2c1, 
        Conv128LayerParam* l4b2c2,
        Fc128LayerParam* bfc1, 
        Out128LayerParam* bout)
{
    grid_group grid = this_grid();
    //========= Conv1 ============
    InConv128Layer(bconv1);
    grid.sync();
    //========= L1B1 ============
    Conv128Layer_new(l1b1c1);
    grid.sync();
    Conv128Layer_new(l1b1c2);
    grid.sync();
    //========= L1B2 ============
    Conv128Layer_new(l1b2c1);
    grid.sync();
    Conv128Layer_new(l1b2c2);
    grid.sync();
    //========= L2B1 ============
    Conv128Layer_new(l2b1c1);
    grid.sync();
    Conv128Layer_new(l2b1c2);
    grid.sync();
    //========= L2B2 ============
    Conv128Layer_new(l2b2c1);
    grid.sync();
    Conv128Layer_new(l2b2c2);
    grid.sync();
    //========= L3B1 ============
    Conv128Layer_new(l3b1c1);
    grid.sync();
    Conv128Layer_new(l3b1c2);
    grid.sync();
    //========= L3B2 ============
    Conv128Layer_new(l3b2c1);
    grid.sync();
    Conv128Layer_new(l3b2c2);
    grid.sync();
    //========= L4B1 ============
    Conv128Layer_new(l4b1c1);
    grid.sync();
    Conv128Layer_new(l4b1c2);
    grid.sync();
    //========= L4B2 ============
    Conv128Layer_new(l4b2c1);
    grid.sync();
    Conv128Layer_new(l4b2c2);
    grid.sync();
    //========= Fc1 ============
    Fc128Layer_new(bfc1);
    grid.sync();
    //========== Output ===========
    Out128Layer_new(bout);
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

    //=============== Get Input and Label =================
    float* images = (float*)malloc(batch*image_height*image_width*image_channel*sizeof(float));
    unsigned* image_labels = (unsigned*)malloc(batch*sizeof(unsigned));
//     read_ImageNet_normalized("./imagenet_files.txt", images, image_labels, batch);

    //================ Get Weight =================
    FILE* config_file = fopen("./resnet_imagenet.csv","r");

    //================ Set Network =================
    //Layer-0
    InConv128LayerParam* bconv1 = new InConv128LayerParam("Conv1", image_height, image_width, 
            7, 7, 3, 64, batch, act_bit, w_bit, 4,4,true,1,1,true);//save residual 
    InConv128LayerParam* bconv1_gpu = bconv1->initialize(images, config_file);

    //Layer-1, basic-block-1, conv1
    Conv128LayerParam* l1b1c1 = new Conv128LayerParam("L1B1C1", bconv1->output_height, 
            bconv1->output_width, 3, 3, 64, 64, batch, act_bit, w_bit);
    Conv128LayerParam* l1b1c1_gpu = l1b1c1->initialize(config_file, bconv1->get_output_gpu());

    //Layer-1, basic-block-1, conv2
    Conv128LayerParam* l1b1c2 = new Conv128LayerParam("L1B1C2", l1b1c1->output_height, 
            l1b1c1->output_width, 3, 3, 64, 64, batch,act_bit, w_bit, 1,1,true,1,1,false,true,true,64);
    Conv128LayerParam* l1b1c2_gpu = l1b1c2->initialize(config_file, l1b1c1->get_output_gpu(),
            bconv1->get_output_residual_gpu());

    //Layer-1, basic-block-2, conv1
    Conv128LayerParam* l1b2c1 = new Conv128LayerParam("L1B2C1", l1b1c2->output_height, 
            l1b1c2->output_width, 3, 3, 64, 64, batch, act_bit, w_bit);
    Conv128LayerParam* l1b2c1_gpu = l1b2c1->initialize(config_file, l1b1c2->get_output_gpu());

    //Layer-1, basic-block-2, conv2
    Conv128LayerParam* l1b2c2 = new Conv128LayerParam("L1B2C2", l1b2c1->output_height, 
            l1b2c1->output_width, 3, 3, 64, 64, batch,act_bit, w_bit, 1,1,true,1,1,false,true,true,128);
    Conv128LayerParam* l1b2c2_gpu = l1b2c2->initialize(config_file, l1b2c1->get_output_gpu(),
            l1b1c2->get_output_residual_gpu());

    //=============
    //Layer-2, basic-block-1, conv1
    Conv128LayerParam* l2b1c1 = new Conv128LayerParam("L2B1C1", l1b2c2->output_height, 
            l1b2c2->output_width, 3, 3, 64, 128, batch, act_bit, w_bit, 2, 2);
    Conv128LayerParam* l2b1c1_gpu = l2b1c1->initialize(config_file, l1b2c2->get_output_gpu());

    //Layer-2, basic-block-1, conv2
    Conv128LayerParam* l2b1c2 = new Conv128LayerParam("L2B1C2", l2b1c1->output_height, 
            l2b1c1->output_width, 3, 3, 128, 128, batch,act_bit, w_bit, 1,1,true,1,1,false,true,true,128,true);
    Conv128LayerParam* l2b1c2_gpu = l2b1c2->initialize(config_file, l2b1c1->get_output_gpu(),
            l1b2c2->get_output_residual_gpu());

    //Layer-2, basic-block-2, conv1
    Conv128LayerParam* l2b2c1 = new Conv128LayerParam("L2B2C1", l2b1c2->output_height, 
            l2b1c2->output_width, 3, 3, 128, 128, batch, act_bit, w_bit, 1, 1);
    Conv128LayerParam* l2b2c1_gpu = l2b2c1->initialize(config_file, l2b1c2->get_output_gpu());

    //Layer-2, basic-block-2, conv2
    Conv128LayerParam* l2b2c2 = new Conv128LayerParam("L2B2C2", l2b2c1->output_height, 
            l2b2c1->output_width, 3, 3, 128, 128, batch,act_bit, w_bit, 1,1,true,1,1,false,true,true,128);
    Conv128LayerParam* l2b2c2_gpu = l2b2c2->initialize(config_file, l2b2c1->get_output_gpu(),
            l2b1c2->get_output_residual_gpu());

    //=============
    //Layer-3, basic-block-1, conv1
    Conv128LayerParam* l3b1c1 = new Conv128LayerParam("L3B1C1", l2b2c2->output_height, 
            l2b2c2->output_width, 3, 3, 128, 256, batch, act_bit, w_bit, 2, 2);
    Conv128LayerParam* l3b1c1_gpu = l3b1c1->initialize(config_file, l2b2c2->get_output_gpu());

    //Layer-3, basic-block-1, conv2
    Conv128LayerParam* l3b1c2 = new Conv128LayerParam("L3B1C2", l3b1c1->output_height, 
            l3b1c1->output_width, 3, 3, 256, 256, batch,act_bit, w_bit, 1,1,true,1,1,false,true,true,128,true);
    Conv128LayerParam* l3b1c2_gpu = l3b1c2->initialize(config_file, l3b1c1->get_output_gpu(),
            l2b2c2->get_output_residual_gpu());

    //Layer-3, basic-block-2, conv1
    Conv128LayerParam* l3b2c1 = new Conv128LayerParam("L3B2C1", l3b1c2->output_height, 
            l3b1c2->output_width, 3, 3, 256, 256, batch, act_bit, w_bit, 1, 1);
    Conv128LayerParam* l3b2c1_gpu = l3b2c1->initialize(config_file, l3b1c2->get_output_gpu());

    //Layer-3, basic-block-2, conv2
    Conv128LayerParam* l3b2c2 = new Conv128LayerParam("L3B2C2", l3b2c1->output_height, 
            l3b2c1->output_width, 3, 3, 256, 256, batch,act_bit, w_bit, 1,1,true,1,1,false,true,true,256);
    Conv128LayerParam* l3b2c2_gpu = l3b2c2->initialize(config_file, l3b2c1->get_output_gpu(),
            l3b1c2->get_output_residual_gpu());

    //=============
    //Layer-4, basic-block-1, conv1
    Conv128LayerParam* l4b1c1 = new Conv128LayerParam("L4B1C1", l3b2c2->output_height, 
            l3b2c2->output_width, 3, 3, 256, 512, batch, act_bit, w_bit, 2, 2);
    Conv128LayerParam* l4b1c1_gpu = l4b1c1->initialize(config_file, l3b2c2->get_output_gpu());

    //Layer-4, basic-block-1, conv2
    Conv128LayerParam* l4b1c2 = new Conv128LayerParam("L4B1C2", l4b1c1->output_height, 
            l4b1c1->output_width, 3, 3, 512, 512, batch,act_bit, w_bit, 1,1,true,1,1,false,true,true,256,true);
    Conv128LayerParam* l4b1c2_gpu = l4b1c2->initialize(config_file, l4b1c1->get_output_gpu(),
            l3b2c2->get_output_residual_gpu());

    //Layer-4, basic-block-2, conv1
    Conv128LayerParam* l4b2c1 = new Conv128LayerParam("L4B2C1", l4b1c2->output_height, 
            l4b1c2->output_width, 3, 3, 512, 512, batch, act_bit, w_bit, 1, 1);
    Conv128LayerParam* l4b2c1_gpu = l4b2c1->initialize(config_file, l4b1c2->get_output_gpu());

    //Layer-4, basic-block-2, conv2
    Conv128LayerParam* l4b2c2 = new Conv128LayerParam("L4B2C2", l4b2c1->output_height, 
            l4b2c1->output_width, 3, 3, 512, 512, batch,act_bit, w_bit, 1,1,true,1,1,true,false,true,512);
    Conv128LayerParam* l4b2c2_gpu = l4b2c2->initialize(config_file, l4b2c1->get_output_gpu(),
            l4b1c2->get_output_residual_gpu());

    //=============
    //Layer-5
    Fc128LayerParam* bfc1 = new Fc128LayerParam("Fc1", batch, (l4b2c2->output_height)
            *(l4b2c2->output_width)*512, 512, act_bit, w_bit); 
    Fc128LayerParam* bfc1_gpu = bfc1->initialize(config_file, l4b2c2->get_output_gpu());
    //Out Layer
    Out128LayerParam* bout = new Out128LayerParam("Fout", batch, 512, output_size, act_bit, w_bit);
    Out128LayerParam* bout_gpu = bout->initialize(config_file, bfc1->get_output_gpu());  

    //================ Setup Kernel =================
    int numThreads = 128;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    int shared_memory = 512*sizeof(int)*32;
    cudaFuncSetAttribute(resnet128, cudaFuncAttributeMaxDynamicSharedMemorySize,shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, resnet128, numThreads, shared_memory);

    void* args[] = {&bconv1_gpu, 
        &l1b1c1_gpu, 
        &l1b1c2_gpu,
        &l1b2c1_gpu,
        &l1b2c2_gpu,
        &l2b1c1_gpu, 
        &l2b1c2_gpu,
        &l2b2c1_gpu,
        &l2b2c2_gpu,
        &l3b1c1_gpu, 
        &l3b1c2_gpu,
        &l3b2c1_gpu,
        &l3b2c2_gpu,
        &l4b1c1_gpu, 
        &l4b1c2_gpu,
        &l4b2c1_gpu,
        &l4b2c2_gpu,
        &bfc1_gpu,
        &bout_gpu};

    START_TIMER;
    for (int i=0; i<NUM_PROF; i++)
    cudaLaunchCooperativeKernel((void*)resnet128, numBlocksPerSm*deviceProp.multiProcessorCount, 
            numThreads, args, shared_memory);

    CUDA_CHECK_KERNEL();
    STOP_TIMER;
    printf("Time: %.3f (ms)\n", milliseconds/NUM_PROF);

    //================ Output =================
//     float* output = bout->download_output();
    //validate_prediction(output, image_labels, output_size, batch);


/*
    float* out = l1b2c1->download_full_output();
    //float* out = l1b1c2->download_full_output();
    //for (int i=0; i<512; i++)
    for (int i=4096; i<4096+512; i++)
    {
        printf("%.f ", out[i]);
        if ((i+1)%32==0) printf("\n");
    }
    printf("\n===%f===\n", bout->bn_scale[0]);
*/ 

    delete bconv1;
    delete l1b1c1;
    delete l1b1c2;
    delete l1b2c1;
    delete l1b2c2;

    delete l2b1c1;
    delete l2b1c2;
    delete l2b2c1;
    delete l2b2c2;

    delete l3b1c1;
    delete l3b1c2;
    delete l3b2c1;
    delete l3b2c2;

    delete l4b1c1;
    delete l4b1c2;
    delete l4b2c1;
    delete l4b2c2;

    delete bfc1;
    delete bout;

    return 0;
}