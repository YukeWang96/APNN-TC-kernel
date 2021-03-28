NVCC = nvcc
# NVCC_FLAG = -std=c++11 -O3 -w  -arch=sm_86 -maxrregcount=64 -rdc=true 
NVCC_FLAG = -std=c++11 -O3 -w  -arch=sm_86 -rdc=true 
LIBS = -Inew_GEMM/ -Inew_CONV/ -IArbitraryPrecision/src/Common -ljpeg 
DEPS = param.h kernel.cuh data.h data.cpp utility.h validation.h
# For debug
#NVCC_FLAG = -std=c++11 -w -O0 -g -G -arch=sm_75 -maxrregcount=64 -rdc=true -Xptxas -v

all: 3_mnist_mlp 4_cifar10_vgg 4_cifar10_resnet \
		5_imagenet_alexnet 5_imagenet_vgg 5_imagenet_resnet 

3_mnist_mlp: 0_mnist_mlp.cu $(DEPS)
	$(NVCC) $(NVCC_FLAG) -o $@ 0_mnist_mlp.cu data.cpp $(LIBS)

4_cifar10_vgg: 1_cifar10_vgg.cu $(DEPS)
	$(NVCC) $(NVCC_FLAG) -o $@ 1_cifar10_vgg.cu data.cpp $(LIBS)

4_cifar10_resnet: 1_cifar10_resnet.cu $(DEPS)
	$(NVCC) $(NVCC_FLAG) -o $@ 1_cifar10_resnet.cu data.cpp $(LIBS)

5_imagenet_alexnet: 2_imagenet_alexnet.cu $(DEPS)
	$(NVCC) $(NVCC_FLAG) -o $@ 2_imagenet_alexnet.cu data.cpp $(LIBS)

5_imagenet_vgg: 2_imagenet_vgg.cu $(DEPS)
	$(NVCC) $(NVCC_FLAG) -o $@ 2_imagenet_vgg.cu data.cpp $(LIBS)

5_imagenet_resnet: 2_imagenet_resnet.cu $(DEPS)
	$(NVCC) $(NVCC_FLAG) -o $@ 2_imagenet_resnet.cu data.cpp $(LIBS)

clean:
	rm  3_mnist_mlp 4_cifar10_vgg 4_cifar10_resnet \
		5_imagenet_alexnet 5_imagenet_vgg 5_imagenet_resnet 