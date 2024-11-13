#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define BLOCKSIZE 256


__global__ void Reduction(float* IN, float* OUT, int size) {
	int gindex = threadIdx.x + blockIdx.x*blockDim.x;
	int t = threadIdx.x;
	__shared__ float partialSum[BLOCKSIZE];


	partialSum[t] = IN[gindex];
	
	//printf("%f\n", partialSum[t]);
	
	
    for (int stride = 1; stride <= blockDim.x; stride *= 2){
	     __syncthreads();
	     if ((t % (stride * 2) == 0) && ((t+stride) < blockDim.x)){
	     	partialSum[t]+= partialSum[t+stride];
	     	//for the first iteration, partial sums --> t0, t2, t4...
	    }
	}
	if (threadIdx.x == 0){
		OUT[blockIdx.x] = partialSum[0];
		//printf("blockId: %d out: %f\n", blockIdx.x, partialSum[0]);
    }
}

__global__ void single_thread(float* OUT, int numBlocks) {
	for (int i = 1; i < numBlocks; i++){
		//printf("OUT: %f\n", OUT[i]);
		OUT[i] += OUT[i-1];
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

  float *in, *out, *IN, *OUT;

  in = (float*)malloc(sizeof(float) * size);
  out = (float*)malloc(sizeof(float) * numBlocks);
  cudaMalloc(&IN, sizeof(float)*size);
  cudaMalloc(&OUT, sizeof(float)*numBlocks);


  for (int i = 0; i < size; i++) {
         in[i] = i;
  }

  cudaMemcpy(IN, in, sizeof(float)*size, cudaMemcpyHostToDevice);
  cudaMemcpy(OUT, out, sizeof(float)*numBlocks, cudaMemcpyHostToDevice);


  double t0 = get_clock();
  Reduction<<<numBlocks, BLOCKSIZE>>>(IN, OUT, size);
  single_thread<<<1, 1>>>(OUT, numBlocks);
  cudaMemcpy(out, OUT, sizeof(float)*numBlocks, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  double t1 = get_clock();

  printf("time: %f s \n", (t1-t0));
  printf("sum %f \n", out[numBlocks-1]);
  

  #if 0
  for (int i = 0; i < size; i++){
  	printf("%f\n", out[i]);
  }
  #endif

  cudaFree(IN);
  cudaFree(OUT);
  free(in);
  free(out);
 
  return 0;
}
