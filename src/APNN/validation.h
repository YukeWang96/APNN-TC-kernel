#ifndef VALIDATION_H
#define VALIDATION_H

#ifndef MAX
#define MAX(a,b) ((a)>(b)?(a):(b))
#endif

uin32 quantize_cpu(float val, int bitwidth){
    const int max_val = 10000;
    const int min_val = -max_val;
    if (val > max_val) val = max_val - 1;
    if (val < min_val) val = min_val + 1;
    uin32 quant_output = (val - min_val) * (1 << bitwidth) /(max_val - min_val);
    return quant_output;
}

bool validate_difference(uin32* ref, uin32* computed, int size){
    int incorrect = 0;
    for (int i=0; i<size; i++){
        if (ref[i] != computed[i]){
            printf("ref: %u --- comp: %u\n", ref, computed);
            incorrect += 1;
        }
    }
    printf("incorrect/total = %d/%d\n", incorrect, size);
    return incorrect == 0? true:false;
}



// print out the [INPUT onvolution layer] on specific item and channel
// * input --> [N, C, H, W]
void print_first_item_input(float* image, InConv128LayerParam* p){
    const int item_id = 0;
    const int channel_id = 0;
    
    printf("====First Layer (INPUT) %d, H: %d, W: %d====\n", item_id, p->input_height, p->input_width);
    for (int c=0; c<(p->input_channels); c++){
        printf("\n\n------INPUT-CH: %d--------\n", c);
        // for (int h=0; h< (p->input_height); h++){
            // for (int w=0; w<(p->input_width); w++){
            for (int h=0; h<3; h++){
                for (int w=0; w<3; w++){
                int idx = item_id*(p->input_channels)*(p->input_height)*(p->input_width) + c*(p->input_height)*(p->input_width) + h*(p->input_width) + w;
                printf("%.2f ", image[idx]);
            }
            printf("\n");
        }
    }
    printf("===END FIRST LAYER (INPUT)====\n\n");
}


// print out the [INPUT onvolution layer] on specific item and channel
// * input --> [K, K, O, C]
template <class ParamType>
void print_first_filter(uin32* filter, ParamType* p){
    const int output_ch_id = 0;

    printf("======First Layer (Filter)======\n");
    for (int c=0; c<(p->input_channels); c++){
        printf("----INPUT-CH: %d--------\n", c);
        for (int h=0; h<(p->filter_height); h++){
            for (int w=0; w<(p->filter_width); w++){
                int idx = (h*(p->filter_width) + w)*(p->output_channels)*(p->input_channels) + output_ch_id*(p->input_channels) + c;
                printf("%d ", filter[idx]);
            }
            printf("\n");
        }
    }
    printf("=====END FIRST LAYER FILTER====\n\n");
    printf("pad_h:%d\npad_w:%d\nstride_height:%d\nstride_width:%d\n", p->pad_h, p->pad_w, p->stride_height, p->stride_width);
}



// print out the input convolution layer OUTPUT for certain item.
// * output --> [H, W, N, O] Float
void print_first_item_output_float(float* image, InConv128LayerParam* p){
    const int item_id = 0;
    const int channel_id = 0;

    printf("\n\n====First Layer (OUTPUT) item:%d,CH:%d=====\n", item_id, channel_id);
    // for (int h=0; h<(p->output_height); h++){
        // for (int w=0; w<(p->output_width); w++){
    for (int h=0; h<3; h++){
        for (int w=0; w<3; w++){
            int idx = (h*(p->output_width) + w)*PAD8(p->batch)*(p->output_channels) + item_id*(p->output_channels) + channel_id;
            printf("%.3f ", image[idx]);
        }
        printf("\n");
    }
    printf("===========END First Layer (OUTPUT)=======\n\n");
}

// print out the input convolution layer OUTPUT for certain item.
// // * output --> [H, W, N, O] INT32
// [N, H, W, O]
template <class ParamType>
void print_first_item_output_uin32(uin32* image, ParamType* p){
    const int item_id = 0;
    const int channel_id = 0;

    printf("\n\n====First Layer (OUTPUT) item:%d,CH:%d=====\n", item_id, channel_id);
    // for (int h=0; h<(p->output_height); h++){
        // for (int w=0; w<(p->output_width); w++){
    for (int h=0; h<3; h++){
        for (int w=0; w<3; w++){
            int idx = item_id*(p->output_height)*(p->output_width)*(p->output_channels)
                    + h*(p->output_width)*(p->output_channels)
                    + w*(p->output_channels) + channel_id;
            printf("%d ", image[idx]);
        }
        printf("\n");
    }
    printf("===========END First Layer (OUTPUT)=======\n\n");
}

// perform a CPU version of GEMM.
template <class ParamType>
uin32* perform_dummy_FC_Layer(uin32* full_input, uin32* full_weight, ParamType* p){
    
    uin32* output= NULL;
    SAFE_ALOC_HOST(output, p->output_size()*sizeof(uin32));
    memset(output, 0, p->output_size()*sizeof(uin32));

    for (int i = 0; i<(p->output_height); i++){
        for (int j = 0; j<(p->output_width); j++){
            int tmp = 0;
            for (int k = 0; k < (p->input_width); k++){
                // printf("%d -- %d\n", full_input[i*(p->input_width)+k], full_weight[j*(p->weight_height)+k]);
                tmp += full_input[i*(p->input_width)+k] * full_weight[j*(p->weight_height)+k];
            }
            output[i*(p->output_width)+j] = tmp;
            // output[i*(p->output_width)+j] = quantize_cpu(tmp, p->act_bit);
            // printf("%d \n", tmp);
        }
    }
    return output;
}

// perform a CPU version of GEMM.
template <class ParamType>
uin32* perform_dummy_FC_inputlayer(uin32* full_input, uin32* full_weight, ParamType* p){
    
    uin32* output= NULL;
    SAFE_ALOC_HOST(output, p->output_size()*sizeof(uin32));
    memset(output, 0, p->output_size()*sizeof(uin32));

    for (int i = 0; i<(p->output_height); i++){
        for (int j = 0; j<(p->output_width); j++){
            int tmp = 0;
            for (int k = 0; k < (p->input_width); k++){
                // printf("%d -- %d\n", full_input[i*(p->input_width)+k], full_weight[j*(p->weight_height)+k]);
                tmp += full_input[i*(p->input_width)+k] * full_weight[j*(p->weight_height)+k];
            }
            // output[i*(p->output_width)+j] = tmp;
            output[i*(p->output_width)+j] = quantize_cpu(tmp, p->act_bit);
        }
    }
    return output;
}


// perform a CPU version of convolution for the Input Layer.
uin32* perform_dummy_convolution_inputLayer(float* input, uin32* filter, InConv128LayerParam* p){
    /*
    * featureMap --> [N, C, H, W], C = 3.
    * filter --> [K_h, K_w, O, C] 
    // * output --> [H, W, N, C] 
    * output --> [N, H, W, C] 
    */
    // uin32* output= NULL;
    // SAFE_ALOC_HOST(output, p->output_size()*sizeof(uin32));
    // memset(output, 0, p->output_size()*sizeof(uin32));

    uin32* output= NULL;
    SAFE_ALOC_HOST(output, p->output_size()*sizeof(uin32));
    memset(output, 0, p->output_size()*sizeof(uin32));

    float* output_tmp= NULL;
    SAFE_ALOC_HOST(output_tmp, p->output_size()*sizeof(float));
    memset(output_tmp, 0, p->output_size()*sizeof(float));

    const int act_bit = 3;

    const int src_output_height = (p->pool_height)*(p->output_height);  // input layer  (p->pool_height) == 1
    const int src_output_width = (p->pool_width)*(p->output_width);     // input layer (p->pool_width) == 1

    printf("input_h: %d, input_w: %d\n", p->input_height, p->input_width);
    printf("output_h: %d, output_w: %d\n", p->output_height, p->output_width);

    // (x, y, bid, c_out) in the output across all output channels, 
    // where c_out is the iteration direction for one (bid) iteration.
    for (int bid = 0; bid < src_output_height*src_output_width*(p->batch); bid++)
    {
        // Get the position in the output feature map. (N, H, W) = (bz, by, bx)
        const int bz = bid/(src_output_width*src_output_height); //over N: batch item id
        const int by = (bid%(src_output_width*src_output_height))/(src_output_width);//over P:out_height
        const int bx = (bid%(src_output_width*src_output_height))%(src_output_width);//over Q:out_width 

        const int dst_y = by/(p->pool_height); // height index (y) position of output. 
        const int dst_x = bx/(p->pool_width);  // width index (x) position of output.

        //coord (ax,ay) in [Input] from bx,by in [Output].
        const int ax0 = bx*(p->stride_width)-(p->pad_w);
        const int ay0 = by*(p->stride_height)-(p->pad_h);

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
                        float f0 = input[(bz*3+0)*(p->input_height)*(p->input_width) + ay*(p->input_width)+ax];//R
                        float f1 = input[(bz*3+1)*(p->input_height)*(p->input_width) + ay*(p->input_width)+ax];//G
                        float f2 = input[(bz*3+2)*(p->input_height)*(p->input_width) + ay*(p->input_width)+ax];//B
                        
                        // if (ay == ax && ax == 0){
                        //     printf("reference --- input@[0, 0], R: %.3f, G:%.3f, B:%.3f\n", f0, f1, f2);
                        // }

                        // [H, W, C, O]
                        for (int k=0; k<(p->output_channels); k++)
                        {
                            uin32 l0 = filter[(r*(p->filter_width)+s)*(p->input_channels)*(p->output_channels) + 0*(p->output_channels) + k];
                            uin32 l1 = filter[(r*(p->filter_width)+s)*(p->input_channels)*(p->output_channels) + 1*(p->output_channels) + k];
                            uin32 l2 = filter[(r*(p->filter_width)+s)*(p->input_channels)*(p->output_channels) + 2*(p->output_channels) + k];
                            
                            // To shape[input_height, input_width, batch, in_channels/32]
                            const int idx = bz*(p->output_height)*(p->output_width)*(p->output_channels)
                                            +dst_y*(p->output_width)*(p->output_channels) //P
                                            +dst_x*(p->output_channels); //Q

                            // const int idx = (dst_y*(p->output_width)*PAD8(p->batch) //P
                            //                 +dst_x*PAD8(p->batch) //Q
                            //                 +bz)*(p->output_channels);

                            // one target point in the k-th output channel.
                            output_tmp[idx + k] += l0 * f0 + l1 * f1 + l2 * f2;
                        } // END for (k)
                    } // END if ((ax>=0) && (ax<(p->input_width)) )
                } // END for (s)
            } // END if ((ay>=0) && (ay<(p->input_height))) 
        } // END for (r)

        // quantize output.
        for (int k=0; k<(p->output_channels); k++)
        {
            // To shape[input_height, input_width, batch, in_channels/32]
            // const int idx = (dst_y*(p->output_width)*PAD8(p->batch) //P
            //                 +dst_x*PAD8(p->batch) //Q
            //                 +bz)*(p->output_channels);

            // [N, H, W, C]
            const int idx = bz*(p->output_height)*(p->output_width)*(p->output_channels)
                            +dst_y*(p->output_width)*(p->output_channels) //P
                            +dst_x*(p->output_channels); //Q

            // printf("%.3f\n", output_tmp[idx+k]);
            output[idx+k] = quantize_cpu(output_tmp[idx+k], act_bit);        
        }
    } // END for (bid)
    return output;
}

// perform a CPU version of convolution for the Hidden Layer.
uin32* perform_dummy_convolution_hiddenLayer(uin32* featureMap, uin32* filter, Conv128LayerParam* p){
    /*
    * featureMap --> [H, W, N, C]
    * filter --> [K_h, K_w, O, C] 
    * output --> [H, W, N, C]
    */ 
    uin32* output= NULL;
    SAFE_ALOC_HOST(output, p->output_size()*sizeof(uin32));
    memset(output, 0, p->output_size()*sizeof(uin32));

    const int act_bit = p->act_bit;
    const int w_bit = p->w_bit;
    const int src_output_height = (p->pool_height)*(p->output_height);//32
    const int src_output_width = (p->pool_width)*(p->output_width);//32

    printf("input_h: %d\ninput_w: %d\n", p->input_height, p->input_width);
    printf("stride_height: %d\nstride_width: %d\n", p->stride_height, p->stride_width);
    printf("pad_h: %d\npad_w: %d\n", p->pad_h, p->pad_w);
    printf("src_output_height: %d\nsrc_output_width: %d\n", src_output_height, src_output_width);
    printf("output_h: %d\noutput_w: %d\n", p->output_height, p->output_width);

    // each block consists of 32x32 threads, 32 warps
    // each warp manage 8 points in the output feature map.
    // /32 * /8 =  /256 = /8*8*4
    for (int bid = 0; bid < src_output_height*src_output_width*(p->output_channels)*(p->batch); bid++)
    {
        // get the position of a point in the output feature map
        // each warp compute 4 x 8 x 8 output region of one output channel.
        // (by, bx, bz, bo) = (H, W, N, C)
        const int by = bid/(src_output_width*(p->output_channels)*(p->batch)); //P: output_height
        const int bx = (bid%(src_output_width*(p->output_channels)*(p->batch)))/((p->output_channels)*(p->batch)); //Q:output_width
        const int bz = (bid%(src_output_width*(p->output_channels)*(p->batch)))%((p->output_channels)*(p->batch)); //output_channel/32*batch/8
        const int bn = bz / (p->output_channels); 
        const int bo = bz % (p->output_channels);

        //coord (ax,ay) in Input from bx,by in Output
        const int ax0 = bx*(p->stride_width)-(p->pad_w);
        const int ay0 = by*(p->stride_height)-(p->pad_h);

        int test = 0;

        // [N, H, W, O]
        int idx = (bn*(p->output_height)*(p->output_width)*(p->output_channels) 
                + (by/(p->pool_height))*(p->output_width)*(p->output_channels))
                + (bx/(p->pool_width))*(p->output_channels)
                + bo;

        uin32 tmp = 0;
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
                    for (int c=0; c<(p->input_channels); c++)
                    {
                        // input:  [N,H,W,C]
                        // filter: [K,K,C,O]
                        // uin32 input_val = featureMap[(ay*(p->input_width)+ax)*(p->batch)*(p->input_channels) + bn*(p->input_channels) + c];
                        uin32 input_val = featureMap[bn*(p->input_height)*(p->input_width)*(p->input_channels) + (ay*(p->input_width)+ax)*(p->input_channels) + c];
                        uin32 filter_val = filter[(r*(p->filter_width)+s)*(p->output_channels)*(p->input_channels) + c*(p->output_channels) + bo];
                        // if (by == 1 && bx == 0 && bo == 0 && bn == 0){
                        //     test += input_val;
                        // }
                        // if before a FC layer.
                        tmp += input_val*filter_val;
                    } // END: for (c)
                } // <-- END: if ((ay>=0)&&(ay<(p->input_height))&&(ax>=0)&&(ax<(p->input_width)))
            } // <--END: for (s)
        } // <--END: for (r)

        // if (output[idx] != 0) 
        //     printf("see this before\n");
        // [N, H, W, O]
        output[idx] = max(output[idx], quantize_cpu(tmp, act_bit));

        // if (by == 1 && bx == 0 && bo == 0 && bn == 0)
        //     printf("=> test_val [0,0,0,0]: %u\n", test);
    } // END for (bid)
    return output;
}
#endif 