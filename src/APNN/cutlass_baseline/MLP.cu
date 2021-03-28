/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
Please check example 07 and 08 for the basics of tensor op gemm kernels.  On NVIDIA Ampere
architecture, most concept still holds.  The two main differences are

1. NVIDIA Ampere architecture introduces a new series of tensor core instructions (see 
   include/cutlass/arch/mma_sm80.h) which are more efficient on Ampere.

2. NVIDIA Ampere architecture uses cp_async() to build multistage software pipeline to better hide
   latency (see include/cutlass/gemm/threadblock/mma_multistage.h)

Moreover, NVIDIA Ampere architecture starts supporting tfloat32 (see include/cutlass/tfloat32.h)
data types in tensor cores.  One big advantage is that we can load in fp32 data and convert them
implicitly to tf32 inside the GEMM kernel which means no change is needed to accelerate traditional
fp32 data by using NVIDIA Ampere architecture.
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"
#include "config.h"

int run() {

  const int batch = 8;        // batch size
  const int input = 896;      // input layer (round 128)
  const int layer_1 = 1024;   // layer-1
  const int layer_2 = 1024;   // layer-2
  const int layer_3 = 1024;   // layer-3
  const int output = 128;     // output layer (round 128)

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_1(batch, layer_1, input);     // input layer
  cutlass::gemm::GemmCoord problem_size_2(batch, layer_2, layer_1);   // layer2
  cutlass::gemm::GemmCoord problem_size_3(batch, layer_3, layer_2);   // layer3
  cutlass::gemm::GemmCoord problem_size_4(batch, output, layer_3);    // output layer

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> input_ten({batch, input});  // <- Nx(28x28)
  cutlass::HostTensor<ElementInputA, LayoutInputA> hidden_1(problem_size_1.mn());  // <- Bx1024
  cutlass::HostTensor<ElementInputA, LayoutInputA> hidden_2(problem_size_2.mn());  // <- Bx1024
  cutlass::HostTensor<ElementInputA, LayoutInputA> hidden_3(problem_size_3.mn());  // <- Bx1024
  cutlass::HostTensor<ElementInputA, LayoutInputA> output_ten(problem_size_4.mn());  // <- Bx10

  cutlass::HostTensor<ElementInputB, LayoutInputB> w_1({layer_1, input});     // (28x28) x 1024
  cutlass::HostTensor<ElementInputB, LayoutInputB> w_2({layer_2, layer_1});   // 1024 x 1024
  cutlass::HostTensor<ElementInputB, LayoutInputB> w_3({layer_3, layer_2});   // 1024 x 1024
  cutlass::HostTensor<ElementInputB, LayoutInputB> w_4({layer_3, output});    // 1024 x 10

  // cutlass::NumericArrayConverter<cutlass::int4b_t, int32_t>::convert(hidden_1_pre);
  // CUTLASS kernel Reference
  // cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
  // reference kernel

  // Copy data from host to GPU
  input_ten.sync_device();
  hidden_1.sync_device();
  hidden_2.sync_device();
  hidden_3.sync_device();
  output_ten.sync_device();

  // hidden_1_pre.sync_device();
  // hidden_2_pre.sync_device();
  // hidden_3_pre.sync_device();
  // output_pre.sync_device();

  w_1.sync_device();
  w_2.sync_device();
  w_3.sync_device();
  w_4.sync_device();
  // tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments_1{problem_size_1, input_ten.device_ref(), w_1.device_ref(), hidden_1.device_ref(),   hidden_1.device_ref(), {alpha, beta}, split_k_slices};        
  typename Gemm::Arguments arguments_2{problem_size_2, hidden_1.device_ref(),  w_2.device_ref(), hidden_2.device_ref(),   hidden_2.device_ref(), {alpha, beta}, split_k_slices};        
  typename Gemm::Arguments arguments_3{problem_size_3, hidden_2.device_ref(),  w_3.device_ref(), hidden_3.device_ref(),   hidden_3.device_ref(), {alpha, beta}, split_k_slices};        
  typename Gemm::Arguments arguments_4{problem_size_4, hidden_3.device_ref(),  w_4.device_ref(), output_ten.device_ref(), output_ten.device_ref(), {alpha, beta}, split_k_slices};        

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size_1 = Gemm::get_workspace_size(arguments_1);
  size_t workspace_size_2 = Gemm::get_workspace_size(arguments_2);
  size_t workspace_size_3 = Gemm::get_workspace_size(arguments_3);
  size_t workspace_size_4 = Gemm::get_workspace_size(arguments_4);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace_1(workspace_size_1);
  cutlass::device_memory::allocation<uint8_t> workspace_2(workspace_size_2);
  cutlass::device_memory::allocation<uint8_t> workspace_3(workspace_size_3);
  cutlass::device_memory::allocation<uint8_t> workspace_4(workspace_size_4);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op_1;
  Gemm gemm_op_2;
  Gemm gemm_op_3;
  Gemm gemm_op_4;

  // Initialize CUTLASS kernel with arguments and workspace pointer
  cutlass::Status status_1 = gemm_op_1.initialize(arguments_1, workspace_1.get());
  cutlass::Status status_2 = gemm_op_2.initialize(arguments_2, workspace_2.get());
  cutlass::Status status_3 = gemm_op_3.initialize(arguments_3, workspace_3.get());
  cutlass::Status status_4 = gemm_op_4.initialize(arguments_4, workspace_4.get());

  CUTLASS_CHECK(status_1);
  CUTLASS_CHECK(status_2);
  CUTLASS_CHECK(status_3);
  CUTLASS_CHECK(status_4);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // Launch initialized CUTLASS kernel
  for(int trial = 0; trial < NUM_PROFILE; trial++) {

    // status_1 = gemm_op_1();
    // cudaDeviceSynchronize();
    // CUTLASS_CHECK(status_1);
    // // printf("pass 1\n");

    // status_2 = gemm_op_2();
    // CUTLASS_CHECK(status_2);
    // cudaDeviceSynchronize();
    // printf("pass 2\n");

    // status_3 = gemm_op_3();
    // CUTLASS_CHECK(status_3);
    // cudaDeviceSynchronize();
    // // printf("pass 3\n");

    status_4 = gemm_op_4();
    cudaDeviceSynchronize();
    CUTLASS_CHECK(status_4);
    // // printf("pass 4\n");
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  int total_ops = 2 * (batch*input*layer_1 + batch*layer_1*layer_2 + batch*layer_2*layer_3 + batch*layer_3*output);
  printf("CUTLASS-GEMM (%d-bit). TOPS: %4.2f\tTime (ms): %.3f\n", 
                                              BIT_WIDTH, 
                                              static_cast<double>(NUM_PROFILE*(static_cast<double>(total_ops)) /
                                              (milliseconds / 1000.)) / 1e12, 
                                              milliseconds/NUM_PROFILE);
  
  // printf("CUTLASS-GEMM (%d-bit). M: %6d, N: %6d, K: %6d,\tTOPS: %4.2f\tTime (ms): %.2f\n", BIT_WIDTH, M, N, K, 
  //                                               static_cast<double>(NUM_PROFILE*(static_cast<double>(M) * N * K * 2) /
  //                                              (milliseconds / 1000.)) / 1e12, milliseconds/NUM_PROFILE);
    /*
  // Create instantiation for device reference gemm kernel
  cutlass::reference::device::Gemm<ElementInputA,
                                   LayoutInputA,
                                   ElementInputB,
                                   LayoutInputB,
                                   ElementOutput,
                                   LayoutOutput,
                                   ElementComputeEpilogue,
                                   ElementComputeEpilogue> gemm_device;

  // Launch device reference gemm kernel
  gemm_device(problem_size,
              alpha,
              tensor_a.device_ref(),
              tensor_b.device_ref(),
              beta,
              tensor_c.device_ref(),
              tensor_ref_d.device_ref());

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::host::TensorEquals(
    tensor_d.host_view(),
    tensor_ref_d.host_view());

  std::cout << (passed ? "Passed" : "Failed") << std::endl;

  // return (passed ? 0  : -1);*/
  return 0;
}

int main(int argc, char* argv[]) {

  bool notSupported = false;

  // Ampere Tensor Core operations exposed with mma.sync and ldmatrix are first available
  // in CUDA 11.0. 
  //
  // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "Turing Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  return run();
}