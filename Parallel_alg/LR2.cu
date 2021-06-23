#include <stdio.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#define CSC(call) do {              \
    cudaError err = call;           \
    if (err != cudaSuccess) {       \
        fprintf(stderr, "CUDA error in file %s in line %d: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(0);                    \
    }                               \
} while(0)

__global__ void kernel_radix(int *arr, int *radix, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while(idx < n) {
        radix[idx] = (arr[idx] >> k) & 1;
        idx += offset;
    }
}

__global__ void kernel_perm(int *arr, int *radix, int *out, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    int sn = radix[n - 1] + ((arr[n - 1] >> k) & 1);
    while(idx < n) {
        if ((arr[idx] >> k) & 1)
            out[radix[idx] + (n - sn)] = arr[idx];
        else
            out[idx - radix[idx]] = arr[idx];
        idx += offset;
    }
}

void radix_sort(int *dev_arr, int n) {
    int *dev_out, *dev_radix, *temp;
    CSC(cudaMalloc(&dev_out, sizeof(int) * n));
    CSC(cudaMalloc(&dev_radix, sizeof(int) * n));
    int k;
    for(k = 0; k < 32; k++) {
        kernel_radix<<<256, 256>>>(dev_arr, dev_radix, n, k);
        thrust::exclusive_scan(thrust::device, dev_radix, dev_radix + n, dev_radix);
        kernel_perm<<<256, 256>>>(dev_arr, dev_radix, dev_out, n, k);
        temp = dev_arr;
        dev_arr = dev_out;
        dev_out = temp;
    }
}

int main() {
    int i, n = 10000;
    int *dev_arr, *arr = (int *)malloc(sizeof(int) * n);
    for(i = 0; i < n; i++)
        arr[i] = (i * i) % 10000;
    CSC(cudaMalloc(&dev_arr, sizeof(int) * n));
    CSC(cudaMemcpy(dev_arr, arr, sizeof(int) * n, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    float time;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start, 0));

    radix_sort(dev_arr, n);

    CSC(cudaGetLastError());
    CSC(cudaEventRecord(stop, 0));
    CSC(cudaEventSynchronize(stop));
    CSC(cudaEventElapsedTime(&time, start, stop));
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));

    printf("time = %f\n", time);

    CSC(cudaMemcpy(arr, dev_arr, sizeof(int) * n, cudaMemcpyDeviceToHost));
    for(i = n - 100; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");

    CSC(cudaFree(dev_arr));
    free(arr);
    return 0;
}