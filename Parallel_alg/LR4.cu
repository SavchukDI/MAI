#include <stdio.h>

#define CSC(call) do { 				\
	cudaError err = call;			\
	if (err != cudaSuccess) {		\
		fprintf(stderr, "CUDA error in file %s in line %d: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err));	\
		exit(0);					\
	}								\
} while(0)


__global__ void kernel(float *src, float *dst, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < n && idy < n) 
		dst[idx * n + idy] = src[idy * n + idx];
}

__global__ void kernel_shared(float *src, float *dst, int n) {
	__shared__ float buff[32][33];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < n && idy < n)
		buff[threadIdx.x][threadIdx.y] = src[idy * n + idx];
	__syncthreads();
	idx = blockIdx.x * blockDim.x + threadIdx.y;
	idy = blockIdx.y * blockDim.y + threadIdx.x;
	if (idx < n && idy < n)
		dst[idx * n + idy] = buff[threadIdx.y][threadIdx.x];
}


#define _index(i) ((i) + ((i) >> 5))

//_index(31) = 31	
//_index(32) = 33	
//_index(63) = 64
//_index(64) = 66

__global__ void kernel_shared_1d(float *src, float *dst, int n) {
	__shared__ float buff[_index(32 * 32)];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < n && idy < n)
		buff[_index(threadIdx.x * 32 + threadIdx.y)] = src[idy * n + idx];
	__syncthreads();
	idx = blockIdx.x * blockDim.x + threadIdx.y;
	idy = blockIdx.y * blockDim.y + threadIdx.x;
	if (idx < n && idy < n)
		dst[idx * n + idy] = buff[_index(threadIdx.y * 32 + threadIdx.x)];
}

int main() {
	int i, j, n = 1000;
	float *src = (float *)malloc(sizeof(float) * n * n);
	float *dst = (float *)malloc(sizeof(float) * n * n);
	for(i = 0; i < n * n; i++)
		src[i] = i;
	float *dev_src, *dev_dst;

	CSC(cudaMalloc(&dev_src, sizeof(float) * n * n));
	CSC(cudaMalloc(&dev_dst, sizeof(float) * n * n));
	CSC(cudaMemset(dev_dst, 0, sizeof(float) * n * n));
	CSC(cudaMemcpy(dev_src, src, sizeof(float) * n * n, cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	float time;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&stop));
	CSC(cudaEventRecord(start, 0));

	kernel_shared_1d<<< dim3(32, 32), dim3(32, 32) >>>(dev_src, dev_dst, n);

	CSC(cudaGetLastError());
	CSC(cudaEventRecord(stop, 0));
	CSC(cudaEventSynchronize(stop));
	CSC(cudaEventElapsedTime(&time, start, stop));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(stop));

	printf("time = %f\n", time);
	
	CSC(cudaMemcpy(dst, dev_dst, sizeof(float) * n * n, cudaMemcpyDeviceToHost));
	for(i = 0; i < n; i++)		
		for(j = 0; j < n; j++)
			if (src[j * n + i] != dst[i * n + j]) 
				fprintf(stderr, "ERROR!\n"); 
	
	CSC(cudaFree(dev_src));
	CSC(cudaFree(dev_dst));
	free(src);
	free(dst);
	return 0;
}