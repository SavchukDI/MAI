#include "stdio.h"
#include "stdlib.h"

#define CSC(call) do { 				\
	cudaError err = call;			\
	if (err != cudaSuccess) {		\
		fprintf(stderr, "CUDA error in file %s in line %d: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err));	\
		exit(0);					\
	}								\
} while(0)

__global__ void kernel_sort(int* x, bool is_odd, int n) {

	int id = blockIdx.x;

	if (is_odd && ((id * 2 + 1) < n))
		if (x[id * 2] > x[id * 2 + 1]) {
			int X = x[id * 2];
			x[id * 2] = x[id * 2 + 1];
			x[id * 2 + 1] = X;
		}

	if (!is_odd && ((id * 2 + 2) < n))
		if (x[id * 2 + 1] > x[id * 2 + 2]) {
			int X = x[id * 2 + 1];
			x[id * 2 + 1] = x[id * 2 + 2];
			x[id * 2 + 2] = X;
		}
}

int main() {

	int n, i;
	int* arr; 
	int* dev_arr;

	scanf("%d", &n);

	arr = (int*)malloc(sizeof(int) * n);

	for (i = 0; i < n; i++)
		scanf("%d", &arr[i]);

	CSC(cudaMalloc((void**)&dev_arr, sizeof(int) * n));
	CSC(cudaMemcpy(dev_arr, arr, sizeof(int) * n, cudaMemcpyHostToDevice));

	for (i = 0; i < n; i++)
		kernel_sort <<<n / 2, 1>>> (dev_arr, (i % 2 == 0), n);

	CSC(cudaMemcpy(arr, dev_arr, sizeof(int) * n, cudaMemcpyDeviceToHost));
	
	
	for (i = 0; i < n; i++)
		printf("%d ", arr[i]);
	printf("\n");

	CSC(cudaFree(dev_arr));
	free(arr);

	return 0;
}