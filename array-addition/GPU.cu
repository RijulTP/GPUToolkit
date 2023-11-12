#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void vectorAdd(int* a, int* b, int* c)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
    return;

}


int main(){
    // Uses CUDA to use functions that parallely calculates the addition
    int a[] = {1,2,3,4,5,6};
    int b[] = {7,8,9,10,11,12};
    int c[sizeof(a) / sizeof(int)] = {0};

    // Create pointers into the GPU
    int* cudaA = 0;
    int* cudaB = 0;
    int* cudaC = 0;

    // Allocate memory in the GPU
    cudaMalloc(&cudaA,sizeof(a));
    cudaMalloc(&cudaB,sizeof(b));
    cudaMalloc(&cudaC,sizeof(c));
    

    // Copy the vectors into the gpu
    cudaMemcpy(cudaA, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, b, sizeof(b), cudaMemcpyHostToDevice);

    vectorAdd <<<1, sizeof(a) / sizeof(a[0])>>> (cudaA, cudaB, cudaC);


    cudaMemcpy(c,cudaC,sizeof(c),cudaMemcpyDeviceToHost);

    for (int i = 0; i < sizeof(c) / sizeof(int); i++)
    {
        printf("c[%d] = %d\n", i, c[i]);
    }

    return;
}