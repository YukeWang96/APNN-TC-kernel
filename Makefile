ARCH = 80

simpleBMMA_CUDA:
	nvcc src/simpleBMMA_CUDA.cu -std=c++11 -O3 -w -rdc=true -arch=sm_$(ARCH) -lcuda -o simpleBMMA_CUDA

simplestBMMA_CUDA:
	nvcc src/simplestBMMA_CUDA.cu -std=c++11 -O3 -w -rdc=true -arch=sm_$(ARCH) -lcuda -o simplestBMMA_CUDA

clean:
	rm -f simpleBMMA_CUDA simplestBMMA_CUDA