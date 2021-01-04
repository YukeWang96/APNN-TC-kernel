ARCH = 80

simpleBMMA_CUDA:
	nvcc src/simpleBMMA_CUDA.cu -std=c++11 -O3 -w -rdc=true -arch=sm_$(ARCH) -lcuda -o simpleBMMA_CUDA

simplestBMMA_CUDA:
	nvcc src/simplestBMMA_CUDA.cu -std=c++11 -O3 -w -rdc=true -arch=sm_$(ARCH) -lcuda -o simplestBMMA_CUDA

simplestBMMA_SASS:
	python -m turingas.main -i sass/simplestBMMA.sass -o cubin/simplestBMMA_SASS.cubin -arch $(ARCH)
	nvcc -g -G -O0 src/simplestBMMA_SASS.cu -lcuda -arch=sm_$(ARCH) -o simplestBMMA_SASS

clean:
	rm -f simpleBMMA_CUDA simplestBMMA_CUDA simplestBMMA_SASS cubin/simplestBMMA_SASS.cubin