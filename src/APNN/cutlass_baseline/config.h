#ifndef CONFIG_H
#define CONFIG_H

#define BIT_WIDTH 4
#define NUM_PROFILE 1000

#if BIT_WIDTH == 32
  typedef float input_t;
  typedef float output_t;
#elif BIT_WIDTH == 16
  typedef cutlass::half_t input_t;
  typedef cutlass::half_t output_t;
#elif BIT_WIDTH == 8
  typedef int8_t input_t;
  typedef int32_t output_t;
#elif BIT_WIDTH == 4
  typedef cutlass::uint4b_t input_t;
  typedef cutlass::uint4b_t output_t;
#elif BIT_WIDTH == 1
  typedef cutlass::uint1b_t input_t;
  typedef int32_t output_t;
#endif

using ElementInputA = input_t;                        // <- data type of elements in input matrix A
using ElementInputB = input_t;                        // <- data type of elements in input matrix B
using ElementOutput = output_t;                        // <- data type of elements in output matrix D
using ElementAccumulator = int32_t;                   // <- data type of accumulator
using ElementComputeEpilogue = output_t;               // <- data type of epilogue operations
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

//-------------full precision CUDA core (PASS) --------------------
#if BIT_WIDTH == 32

using Element = float;

using Gemm = cutlass::gemm::device::Gemm<
  Element, 
  cutlass::layout::RowMajor,
  Element, 
  cutlass::layout::ColumnMajor,
  Element,
  cutlass::layout::RowMajor, 
  Element,
  cutlass::arch::OpClassSimt, 
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<32, 64, 8>,
  cutlass::gemm::GemmShape<32, 64, 8>, 
  cutlass::gemm::GemmShape<1, 1, 1>,
  cutlass::epilogue::thread::LinearCombination<
      Element, 
      1,
      Element, 
      Element>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
  4
>;


//-------------half precision Tensor core (PASS) --------------------
#elif BIT_WIDTH == 16

using ElementOutput = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;

using Gemm = cutlass::gemm::device::Gemm<
  cutlass::half_t,
  cutlass::layout::RowMajor,
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  ElementOutput,
  cutlass::layout::RowMajor,
  ElementAccumulator,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<64, 64, 64>,
  cutlass::gemm::GemmShape<64, 32, 32>,
  cutlass::gemm::GemmShape<16, 8, 8>,
  cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    64 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  2
>;

//-------------INT-8 Tensor core (PASS) --------------------
#elif BIT_WIDTH == 8

using ElementOutput = int32_t;
using ElementAccumulator = int32_t;
using ElementCompute = int32_t;

using Gemm = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor, 
    int8_t, cutlass::layout::ColumnMajor, 
    ElementOutput, cutlass::layout::RowMajor,
    ElementAccumulator, 
    cutlass::arch::OpClassTensorOp, 
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>, 
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombinationClamp<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 6>;


//-------------INT-4 Tensor core (PASS) --------------------
#elif BIT_WIDTH == 4

using Gemm = cutlass::gemm::device::Gemm<
  cutlass::uint4b_t, cutlass::layout::RowMajor,
  cutlass::uint4b_t, cutlass::layout::ColumnMajor,
  cutlass::uint4b_t, cutlass::layout::RowMajor,
  int32_t,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128, 128, 256>,
  cutlass::gemm::GemmShape<64, 64, 256>,
  cutlass::gemm::GemmShape<16, 8, 64>,
  cutlass::epilogue::thread::LinearCombinationClamp<
    cutlass::uint4b_t,
    8,
    int32_t,
    float
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  2
>;

// using Gemm = cutlass::gemm::device::Gemm<
//   cutlass::int4b_t, cutlass::layout::RowMajor,
//   cutlass::int4b_t, cutlass::layout::ColumnMajor,
//   ElementOutput, cutlass::layout::RowMajor,
//   ElementAccumulator,
//   cutlass::arch::OpClassTensorOp,
//   cutlass::arch::Sm80,
//   cutlass::gemm::GemmShape<128, 128, 256>,
//   cutlass::gemm::GemmShape<64, 64, 256>,
//   cutlass::gemm::GemmShape<16, 8, 64>,
//   cutlass::epilogue::thread::LinearCombinationClamp<
//     ElementOutput,
//     128 / cutlass::sizeof_bits<ElementOutput>::value,
//     ElementAccumulator,
//     ElementCompute
//   >,
//   cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
//   2
// >;

//-------------INT-1 Tensor core (PASS)--------------------
#elif BIT_WIDTH == 1
    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = int32_t;

    using Gemm = cutlass::gemm::device::Gemm<
    cutlass::uint1b_t, cutlass::layout::RowMajor, 
    cutlass::uint1b_t, cutlass::layout::ColumnMajor, 
    ElementOutput, cutlass::layout::RowMajor,
    ElementAccumulator, 
    cutlass::arch::OpClassTensorOp, 
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 512>,
    cutlass::gemm::GemmShape<64, 64, 512>,
    cutlass::gemm::GemmShape<8, 8, 128>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 128, 128,
    false, cutlass::arch::OpXorPopc>;
#endif

#endif // CONFIG_H