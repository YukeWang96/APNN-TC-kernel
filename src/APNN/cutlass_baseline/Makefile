INC=-I/home/yuke/cutlass/include -I/home/yuke/cutlass/tools/util/include -I/home/yuke/cutlass/examples/common
FLAG= -std=c++11 -O3 -w -arch=sm_86 
# -maxrregcount=64
#  -rdc=true 

# all: GEMM
# all: MLP
all: CONV

MLP: MLP.cu config.h
	nvcc $(INC) $(FLAG) $< -o $@

GEMM: GEMM.cu config.h
	nvcc $(INC) $(FLAG) $< -o $@

CONV: CONV.cu # config.h
	nvcc $(INC) $(FLAG) $< -o $@

clean:
	rm -r MLP CONV GEMM