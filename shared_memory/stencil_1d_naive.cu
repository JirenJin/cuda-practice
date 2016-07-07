#include <stdio.h>

#define BLOCK_SIZE 1024
#define RADIUS 30
#define N 10000000 // size of 1D input array

__global__ void stencil_1d(int *in, int *out) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // detect out-of-bound 
    if (index >= N - 2 * RADIUS){
        return;
    }
    
    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++){
        result += in[index + RADIUS + offset]; 
    }

    // Store the result
    out[index] = result;
}

void random_ints(int *a, int n){
    int i;
    for (i = 0; i < n; ++i)
        //a[i] = rand();
        a[i] = 1;
}


int main(void) {
    int *h_in, *h_out;
    int *d_in, *d_out;
    int in_size = N * sizeof(int);
    int out_size = (N - 2 * RADIUS) * sizeof(int);

    h_in = (int *)malloc(in_size); random_ints(h_in, N);
    h_out = (int *)malloc(out_size);

    // initialize timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void **)&d_in, in_size);
    cudaMalloc((void **)&d_out, out_size);

    cudaMemcpy(d_in, h_in, in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, out_size, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    // Launch stencil_1d() kernel on GPU
    stencil_1d<<<(N - 2 * RADIUS + BLOCK_SIZE - 1) / BLOCK_SIZE,BLOCK_SIZE>>>(d_in, d_out);
    cudaEventRecord(stop);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);

    // synchronization
    cudaDeviceSynchronize();


    // print out the resulting array
    /*
    for (int i =0; i < N - 2 * RADIUS; i++) {
        printf("%d: %d", i, h_out[i]);
        printf(((i % 4) != 3) ? "\t" : "\n");
    }
    */
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("used time: %fms\n", milliseconds);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
