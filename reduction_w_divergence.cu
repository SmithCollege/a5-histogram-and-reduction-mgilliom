#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define BLOCKSIZE 100


__global__ void Reduction(float* X, int size) {
	int gindex = threadIdx.x + blockIdx.x*blockDim.x;
	int t = threadIdx.x;
	__shared__ float partialSum[BLOCKSIZE];


	partialSum[t] = X[gindex];
	
	//printf("%f\n", partialSum[t]);
	
	
    for (int stride = 1; stride <= blockDim.x; stride *= 2){
	     __syncthreads();
	     if ((t % (stride * 2) == 0) && ((t+stride) < blockDim.x)){
	     	partialSum[t]+= partialSum[t+stride];
	     	
	     	//for the first iteration, partial sums --> t0, t2, t4...
	    }
	}
	X[gindex] = partialSum[t];
    
}


double get_clock(){
	struct timeval tv; int ok;
	ok = gettimeofday(&tv, (void *) 0);
	if (ok<0) { printf("gettimeofday error"); }
	return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}


int main(void) {
  int size; //has to be a multiple of tile_width^2
  printf("N: ");
  scanf("%d", &size);

  float *x, *X;

  x = (float*)malloc(sizeof(float) * size);
  cudaMalloc(&X, sizeof(float)*size);


  for (int i = 0; i < size; i++) {
         x[i] = i;
  }

  cudaMemcpy(X, x, sizeof(float)*size, cudaMemcpyHostToDevice);

  int numBlocks = ceil(size/BLOCKSIZE);
  printf("num blocks %d\n", numBlocks);

  double t0 = get_clock();
  Reduction<<<numBlocks, BLOCKSIZE>>>(X, size);
  cudaMemcpy(x, X, sizeof(float)*size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  double t1 = get_clock();

  printf("time: %f s \n", (t1-t0));
  printf("sum %f \n", x[0]);
  

  #if 0
  for (int i = 0; i < size; i++){
  	printf("%f\n", x[i]);
  }
  #endif

  cudaFree(X);
  free(x);
 
  return 0;
}
