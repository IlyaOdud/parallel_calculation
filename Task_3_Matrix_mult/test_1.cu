#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cublas_v2.h"
#include <iostream>

void Rand_matrix(int n, double *Matrix){
    double max = 1.0;
    double min = -1.0;
    srand((unsigned) time(NULL));
    for (int i=0; i<n; i++){
        for (int j=0; j<n; j++){
            Matrix[i*n + j] = min + (double)rand()*(max - min)/(double)RAND_MAX;
        }
    }
}

void Matrix_multiplication_host(int n, double *A, double *B, double *C){
    double sum = 0;
    for (int i=0; i<n; i++){
        for (int j=0; j<n; j++){
            sum = 0;
            for (int k=0; k<n; k++){
                sum += A[i*n + k]*B[k*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

__global__ void Matrix_multiplication_device_2(int n, double *A, double *B, double *C){
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    double sum = 0.0;
    for (int k=0; k<n; k++){
        sum += A[index_x*n + k]*B[k*n + index_y];
    }
    C[index_x*n + index_y] = sum;
}

__global__ void Matrix_multiplication_device_1(int n, double *A, double *B, double *C){
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    double sum = 0.0;
     for (int k=0; k<n; k++){
        sum += A[index_y*n + k]*B[k*n + index_x];
    }
    C[index_y*n + index_x] = sum;
}

int main(void){
    const int n = 1024;
    dim3 threads = dim3(32,32);
    dim3 blocks = dim3(n/threads.x, n/threads.y);
    int size = n * n * sizeof(double);
    double *A, *B, *C, *C_test;
    double *A_dev, *B_dev, *C_dev;
    double diff, diff_max, diff_abs_sum;
    //cublasHandle_t handle;
    //double al = 0.0;
    //double bet = 0.0;

    A = (double*)malloc(size);
    B = (double*)malloc(size);
    C = (double*)malloc(size);
    C_test = (double*)malloc(size);

    cudaMalloc((void**)&A_dev, size); 
    cudaMalloc((void**)&B_dev, size);
    cudaMalloc((void**)&C_dev, size);

    Rand_matrix(n,A);
    Rand_matrix(n,B);

    cudaMemcpy(A_dev, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B, size, cudaMemcpyHostToDevice);

    Matrix_multiplication_device_1<<<blocks, threads>>>(n, A_dev, B_dev, C_dev);
    //Matrix_multiplication_device_2<<<blocks, threads>>>(n, A_dev, B_dev, C_dev);

    //cublasCreate(&handle);
    //cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &al, A_dev, B_dev, n, &bet, C_dev, n);

    cudaMemcpy(C, C_dev, size, cudaMemcpyDeviceToHost);

    Matrix_multiplication_host(n, A, B, C_test);

    diff_max = 0;
    diff_abs_sum = 0;
    for (int i=0; i<n; i++){
        for (int j=0; j<n; j++){
            diff = abs(C[i*n + j] - C_test[i*n + j]);
            diff_abs_sum += diff;
            if (diff > diff_max){
                diff_max = diff;
            }
            else{
                continue;
            }
        }
    }
    std::cout << "Sum " << diff_abs_sum << std::endl;
    std::cout << "Max " << diff_max << std::endl;

    free(A);
    free(B);
    free(C);
    free(C_test);
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    
    return 0;
}