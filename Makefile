simpleBMMA_CUDA:
	nvcc simpleBMMA_CUDA.cu -std=c++11 -O3 -w -rdc=true -arch=sm_75 -lcuda -o simpleBMMA_CUDA

clean:
	rm -f simpleBMMA_CUDA