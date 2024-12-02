#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

void Sum(float* x, int size) {
	for (int i = 1; i < size; i++){
		x[i]+=x[i-1];
	}
}

double get_clock(){
	struct timeval tv; int ok;
	ok = gettimeofday(&tv, (void *) 0);
	if (ok<0) { printf("gettimeofday error"); }
	return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}


int main() {
  int size;
  printf("Size: ");
  scanf("%d", &size);

  float* x = malloc(sizeof(float) * size);
  float sum;

  for (int i = 0; i < size; i++) {
      x[i] = 1;
  }

  double t0 = get_clock();
  Sum(x, size);
  double t1 = get_clock();
  printf("time: %f s\n", (t1-t0));


  printf("sum: %f\n", x[size-1]);

  #if 0
  for (int i = 0; i < size; i++){
  	printf("%f\n", x[i]);
  }
  #endif

  return 0;
}
