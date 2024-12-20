#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define BLOCKSIZE 8


__global__ void Reduction(float* IN, float* OUT, int size) {
	int gindex = threadIdx.x + blockIdx.x*blockDim.x;
	int t = threadIdx.x;
	__shared__ float partialSum[BLOCKSIZE*2];

	if (gindex < size){
		partialSum[t] = IN[blockIdx.x * BLOCKSIZE + t];
		//printf("made it into shared: %f\n", partialSum[t]);
		
	    for (unsigned int stride = BLOCKSIZE / 2; stride >= 1; stride /= 2) {
		     __syncthreads();
		     if ((t < stride) && (gindex+stride < size)){
		     	partialSum[t] += partialSum[t+stride];
		 	 }
		}
		
		if (t == 0){
			OUT[blockIdx.x] = partialSum[0];
			//printf("made it into OUT %f\n", partialSum[0]);
	    }
	}
}

__global__ void single_thread(float* OUT, int numBlocks, float* RESULT) {
	for (int i = 1; i < numBlocks; i++){
		//printf("OUT: %f\n", OUT[i]);
		OUT[i] += OUT[i-1];
	}
	RESULT[0] = OUT[numBlocks-1];
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

  float *in, *out, *IN, *OUT, *result, *RESULT;

  in = (float*)malloc(sizeof(float) * size);
  out = (float*)malloc(sizeof(float) * numBlocks);
  result = (float*)malloc(sizeof(float));
  cudaMalloc(&IN, sizeof(float)*size);
  cudaMalloc(&OUT, sizeof(float)*numBlocks);
  cudaMalloc(&RESULT, sizeof(float));

  result[0] = 0.0;

  for (int i = 0; i < size; i++) {
         in[i] = 1;
         //printf("in[%d] = %f\n", i, in[i]);
  }


  cudaMemcpy(IN, in, sizeof(float)*size, cudaMemcpyHostToDevice);
  cudaMemcpy(OUT, out, sizeof(float)*numBlocks, cudaMemcpyHostToDevice);
  cudaMemcpy(RESULT, result, sizeof(float), cudaMemcpyHostToDevice);


  double t0 = get_clock();
  Reduction<<<numBlocks, BLOCKSIZE>>>(IN, OUT, size);
  single_thread<<<1, 1>>>(OUT, numBlocks, RESULT);
  cudaMemcpy(result, RESULT, sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  double t1 = get_clock();

  printf("time: %f s \n", (t1-t0));
  printf("sum %f \n", result[0]);
  

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
