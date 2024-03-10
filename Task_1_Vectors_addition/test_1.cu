#include <stdio.h>
#include <stdlib.h>

__global__ void add_dev(int *a, int *b, int*c, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n){
        c[index] = a[index] + b[index];
    }
}
void run_ints(int *a, int n){
    srand((unsigned) time(NULL));
    for (int i = 0; i < n; i++){
        a[i] = rand(); 
    }
}

int main(void){
    const int n = 2048 * 2048; 
    const int thread_per_block = 512;
    int *a, *b, *c; 
    int *device_a, *device_b, *device_c; 
    int size = sizeof(int) * n; 
    int nblocks;
    int sum;

    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);
    cudaMalloc((void**)&device_a, size); 
    cudaMalloc((void**)&device_b, size);
    cudaMalloc((void**)&device_c, size);

    run_ints(a, n);
    run_ints(b, n);
    cudaMemcpy (device_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy (device_b, b, size, cudaMemcpyHostToDevice);

    
    nblocks = (n - 1) / thread_per_block + 1;
    add_dev <<< nblocks, thread_per_block >>> (device_a, device_b, device_c, n);
    cudaMemcpy (c, device_c, size, cudaMemcpyDeviceToHost);

    sum = 0;
    for (int i = 0; i < n; i++){
        sum += c[i] - a[i] - b[i];
    }
    if (sum == 0){
        printf("Good\n");
    }
    else{
        printf("Worse\n");
    }
    free(a);
    free(b);
    free(c);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    
    return 0;
}