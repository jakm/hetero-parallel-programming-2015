// Excercise 4 - List Reduction

// MP Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void total(float * input, float * output, int len) {
    //@@ Load a segment of the input vector into shared memory
    //@@ Traverse the reduction tree
    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index

    int t = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE * 2 + t;

    __shared__ float ds_partialSum[BLOCK_SIZE * 2];

    ds_partialSum[t] = i < len ? input[i] : 0.0;
    ds_partialSum[t + BLOCK_SIZE] = i + BLOCK_SIZE < len ? input[i + BLOCK_SIZE] : 0.0;

    for (unsigned int stride = BLOCK_SIZE; stride > 0; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            ds_partialSum[t] += ds_partialSum[t + stride];
        }
    }

    if (t == 0) {
        output[blockIdx.x] = ds_partialSum[0];
    }
}

int getNumOutputElements(int numInputElements) {
    int numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }
    return numOutputElements;
}

int reduce(float *hostInput, float *hostOutput, int numInputElements) {
    float * deviceInput;
    float * deviceOutput;
    int numOutputElements; // number of elements in the output list

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here

    size_t inputSize = numInputElements * sizeof(float);

    wbCheck(cudaMalloc((void **) &deviceInput, inputSize));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here

    wbCheck(cudaMemcpy(deviceInput, hostInput, inputSize, cudaMemcpyHostToDevice));

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    size_t outputSize;

    do {
        numOutputElements = getNumOutputElements(numInputElements);

        wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
        wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

        wbTime_start(GPU, "Allocating GPU memory.");
        //@@ Allocate GPU memory here

        outputSize = numOutputElements * sizeof(float);

        wbCheck(cudaMalloc((void **) &deviceOutput, outputSize));

        wbTime_stop(GPU, "Allocating GPU memory.");

        //@@ Initialize the grid and block dimensions here

        dim3 DimGrid(numOutputElements, 1, 1);
        dim3 DimBlock(BLOCK_SIZE, 1, 1);

        wbTime_start(Compute, "Performing CUDA computation");
        //@@ Launch the GPU Kernel here

        total<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numInputElements);

        cudaDeviceSynchronize();
        wbTime_stop(Compute, "Performing CUDA computation");

        wbTime_start(GPU, "Freeing GPU Memory");
        //@@ Free the GPU memory here

        wbCheck(cudaFree(deviceInput));

        wbTime_stop(GPU, "Freeing GPU Memory");

        deviceInput = deviceOutput;
        numInputElements = numOutputElements;
    } while (numOutputElements > 1);

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here

    wbCheck(cudaMemcpy(hostOutput, deviceOutput, outputSize, cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here

    wbCheck(cudaFree(deviceOutput));

    wbTime_stop(GPU, "Freeing GPU Memory");

    return 0;
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    int numInputElements; // number of elements in the input list
    int exitStatus = EXIT_SUCCESS;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);
    hostOutput = (float*) malloc(sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    int status = reduce(hostInput, hostOutput, numInputElements);

    if (status == -1) {
        exitStatus = EXIT_FAILURE;
        goto OUT;
    }

    wbSolution(args, hostOutput, 1);

    OUT:
    free(hostInput);
    free(hostOutput);

    return exitStatus;
}
