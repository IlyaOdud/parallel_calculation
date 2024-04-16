// Command to compile:
// nvcc -c -I/usr/local/cuda/include cusolv_test.cu 
// g++ -o a.out cusolv_test.o -L/usr/local/cuda/lib64 -lcudart -lcusolver
// https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#define _CUSOLVER_ERR_TO_STR(err) \
  case err:                       \
    return #err;

inline const char *cusolver_error_to_string(cusolverStatus_t err) {
    switch (err) {
        _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_SUCCESS);
        _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_NOT_INITIALIZED);
        _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_ALLOC_FAILED);
        _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_INVALID_VALUE);
        _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_ARCH_MISMATCH);
        _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_EXECUTION_FAILED);
        _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_INTERNAL_ERROR);
        _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
        _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_ZERO_PIVOT);
        _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_NOT_SUPPORTED);
        default:
        return "CUSOLVER_STATUS_UNKNOWN";
    };
    }
    
int main() {
    int64_t n_cols = 55296;

    cusolverDnHandle_t cusolverH = nullptr;
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolverH);
    cusolverDnParams_t dn_params = nullptr;
    cusolverDnCreateParams(&dn_params);

    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    
    float* d_eig_vals, *in;
    cudaMalloc((void**)&d_eig_vals, sizeof(float) * n_cols);
    cudaMalloc((void**)&in, sizeof(float) * n_cols * n_cols);
    cudaMemset((void*)d_eig_vals, 0, sizeof(float) * n_cols);
    cudaMemset((void*)in, 0, sizeof(float) * n_cols * n_cols);

    size_t workspaceDevice = 0;
    size_t workspaceHost = 0;
    cusolver_status = cusolverDnXsyevd_bufferSize(
        cusolverH, dn_params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
        n_cols, CUDA_R_32F, in, n_cols, CUDA_R_32F, d_eig_vals, CUDA_R_32F,
        &workspaceDevice, &workspaceHost);
    printf("Call result: %s\n", cusolver_error_to_string(cusolver_status));
    assert (cudaSuccess == cusolver_status);
    std::cout << workspaceDevice << '\n';

    float *d_work, *h_work = 0;
    int *d_dev_info = 0;
    auto res = cudaMalloc((void**)&d_dev_info, sizeof(int));
    assert (res == 0);
    res = cudaMalloc((void**)&d_work, workspaceDevice);
    assert (res == 0);
    cudaMemset((void*)d_dev_info, 0, sizeof(int));
    cudaMemset((void*)d_work, 0, workspaceDevice);
    h_work = (float*)malloc(workspaceHost);


    cusolver_status = cusolverDnXsyevd(
        cusolverH, dn_params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
        n_cols, CUDA_R_32F, in, n_cols, CUDA_R_32F,
        d_eig_vals, CUDA_R_32F, d_work, workspaceDevice, h_work, workspaceHost,
        d_dev_info);
    printf("Call result: %s\n", cusolver_error_to_string(cusolver_status));
    assert (cudaSuccess == cusolver_status);
    free(h_work);
    cudaFree(d_work);
    cudaFree(d_dev_info);
    cudaFree(d_eig_vals);
    cudaFree(in);
    return 0;
}
