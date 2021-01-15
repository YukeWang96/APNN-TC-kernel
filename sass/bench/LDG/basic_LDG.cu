#include <cuda.h>
#include <stdio.h>
#include <string.h>



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


char* concat(const char *s1, const char *s2)
{
    char *result = (char*)malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
    // in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void run(char * name){
	char * file_name = concat(name, ".cubin");

	int *A;
	cudaMalloc((void**)&A, sizeof(char)*128*1024*1024);

	CUmodule module;
	CUfunction kernel;

	cuModuleLoad(&module, file_name);
	cuModuleGetFunction(&kernel, module, "kern");

	void * args[1] = {&A};
	cuLaunchKernel(kernel, 1, 1, 1,
			32, 1, 1,
			32*1024, 0, args, 0);

}

int main(){
	run("basic_LDG");
	return 0;
}
