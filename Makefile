simpleBMMA_CUDA:
	nvcc simpleBMMA_CUDA.cu -arch=sm_75 -lcuda -std=c++11 -o simpleBMMA_CUDA