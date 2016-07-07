#include <stdio.h>

#define BLOCK_SIZE 512
#define RADIUS 3
#define N 10 // size of 1D input array

__global__ void stencil_1d(int *in, int *out) {
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x + blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    // detect out-of-bound 
    //if (gindex >= N - RADIUS)
        //return;

    // Read input elements into shared memory
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS) {
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + BLOCK_SIZE] = in [gindex + BLOCK_SIZE];
    }
    
    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
        //printf("%d", temp[lindex + offset]);
        result += temp[lindex + offset]; 

    // Store the result
    out[gindex] = result;
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

    for (int i =0; i < N; i++) {
        printf("%d", h_in[i]);
        printf(((i % 4) != 3) ? "\t" : "\n");
    }

    cudaMalloc((void **)&d_in, in_size);
    cudaMalloc((void **)&d_out, out_size);

    cudaMemcpy(d_in, h_in, in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, out_size, cudaMemcpyHostToDevice);

    // Launch stencil_1d() kernel on GPU
    stencil_1d<<<(N - 2 * RADIUS + BLOCK_SIZE - 1) / BLOCK_SIZE,BLOCK_SIZE>>>(d_in, d_out);

    // synchronization
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);


    // print out the resulting array
    for (int i =0; i < N - 2 * RADIUS; i++) {
        printf("%d", h_out[i]);
        printf(((i % 4) != 3) ? "\t" : "\n");
    }

    // Clean up
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
