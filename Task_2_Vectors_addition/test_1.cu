#include <iostream>
#include <stdlib.h>

const int n = 512; /*  n must be less then thread_per_block  */
const int thread_per_block = 512;
const int rad = 3;

__global__ void stencil_ld(int *in, int *out){
    __shared__ int temp[thread_per_block + 2*rad];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + rad;
    temp[lindex] = in[gindex];
    if (threadIdx.x < rad){
        temp[lindex - rad] = in[gindex - rad];
        temp[lindex + thread_per_block] = in[gindex + thread_per_block];
    }
    __syncthreads();
    int result = 0;
    for (int offset = -rad; offset <= rad; offset++){
        result += temp[lindex + offset];
    }
    out[gindex] = result;
}

void run_ints(int *a, int n){
    //srand((unsigned) time(NULL));
    for (int i = 0; i < n; i++){
        a[i] = i;
    }
}

int main(void){
    int *input_vec, *output_vec_1, *output_vec_2;
    int *dev_input_vec, *dev_output_vec;
    int size = sizeof(int) * n;
    int sum;

    input_vec = (int*)malloc(size);
    output_vec_1 = (int*)malloc(size);
    output_vec_2 = (int*)malloc(size);
    cudaMalloc((void**)&dev_input_vec, size);
    cudaMalloc((void**)&dev_output_vec, size);

    run_ints(input_vec, n);

    cudaMemcpy (dev_input_vec, input_vec, size, cudaMemcpyHostToDevice);

    stencil_ld <<< rad, thread_per_block >>> (dev_input_vec, dev_output_vec);
    cudaMemcpy (output_vec_2, dev_output_vec, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++){
        output_vec_1[i] = input_vec[i];
        for(int j = 1; j <= rad; j++){
            if (i - j >= 0){
                output_vec_1[i] += input_vec[i - j];
            }
            if (i + j <= n-1){
                output_vec_1[i] += input_vec[i + j];
            }
        }
    }

    sum = 0;
    for (int i = 0; i < n; i++){
        sum += output_vec_1[i] - output_vec_2[i];
    }
    std::cout << "Difference sum(host - device):\t" << sum << std::endl;

    free(input_vec);
    free(output_vec_1);
    free(output_vec_2);
    cudaFree(dev_input_vec);
    cudaFree(dev_output_vec);
    
    return 0;
}