#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define BLOCKSIZE 256


__global__ void histo_kernel(int *IN, unsigned int *OUT, int size){
     int i = threadIdx.x + blockIdx.x * blockDim.x;
     // stride is total number of threads
     int stride = blockDim.x * gridDim.x;

      // All threads in the grid collectively handle
      // blockDim.x * gridDim.x consecutive elements
      while (i < size) {
	      atomicAdd( &(OUT[IN[i]]), 1);
	      printf("%u \n", OUT[i]);
	      i += stride;
      }
     
}




double get_clock(){
	struct timeval tv; int ok;
	ok = gettimeofday(&tv, (void *) 0);
	if (ok<0) { printf("gettimeofday error"); }
	return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}


int main(void) {
  int size;
  printf("N: ");
  scanf("%d", &size);

  int numBlocks = ceil(1.0 * size/BLOCKSIZE);
  printf("num blocks %d\n", numBlocks);

  int *in, *IN;
  unsigned int *out, *OUT;

  in = (int*)malloc(sizeof(int) * size);
  out = (unsigned int*)malloc(sizeof(unsigned int) * 10);
  cudaMalloc(&IN, sizeof(int)*size);
  cudaMalloc(&OUT, sizeof(unsigned int)*numBlocks);


  for (int i = 0; i < size; i++) {
         in[i] = 2;         
  }
  for (unsigned int i = 0; i < 10; i++) {
         out[i] = 0;         
  }

  cudaMemcpy(IN, in, sizeof(int)*size, cudaMemcpyHostToDevice);
  cudaMemcpy(OUT, out, sizeof(unsigned int)*10, cudaMemcpyHostToDevice);


  double t0 = get_clock();
  histo_kernel<<<numBlocks, BLOCKSIZE>>>(IN, OUT, size);
  cudaGetErrorString(cudaGetLastError());
  cudaMemcpy(out, OUT, sizeof(unsigned int) * 10, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  double t1 = get_clock();

  printf("time: %f s \n", (t1-t0));
  for (int i = 0; i < 10; i++)
  	printf("number: %d count: %u \n", i, out[i]);
  

  cudaFree(IN);
  cudaFree(OUT);
  free(in);
  free(out);
 
  return 0;
}
