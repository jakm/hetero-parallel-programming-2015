// Excercise 3 - Image Convolution

#include <wb.h>
#include <assert.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

//@@ INSERT CODE HERE

#define O_TILE_WIDTH 12
#define BLOCK_WIDTH (O_TILE_WIDTH + Mask_width - 1)

#define NUM_CHANNELS 3
#define CHANNEL_R 0
#define CHANNEL_G 1
#define CHANNEL_B 2

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


__device__ float clamp(float x, float start, float end) {
    return MIN(MAX(x, start), end);
}

__global__ void imageConvolution(float *inputImageData, float *outputImageData,
                                 int imageWidth, int imageHeight,
                                 const float * __restrict__ Mask) {

    __shared__ float ds_imageData[BLOCK_WIDTH][BLOCK_WIDTH][NUM_CHANNELS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y*O_TILE_WIDTH + ty;
    int col_o = blockIdx.x*O_TILE_WIDTH + tx;
    int row_i = row_o - 2;
    int col_i = col_o - 2;

    if (row_i >= 0 && row_i < imageHeight && col_i >= 0 && col_i < imageWidth) {
        ds_imageData[ty][tx][CHANNEL_R] = inputImageData[(row_i*imageWidth + col_i)*NUM_CHANNELS + CHANNEL_R];
        ds_imageData[ty][tx][CHANNEL_G] = inputImageData[(row_i*imageWidth + col_i)*NUM_CHANNELS + CHANNEL_G];
        ds_imageData[ty][tx][CHANNEL_B] = inputImageData[(row_i*imageWidth + col_i)*NUM_CHANNELS + CHANNEL_B];
    } else {
        ds_imageData[ty][tx][CHANNEL_R] = 0.0;
        ds_imageData[ty][tx][CHANNEL_G] = 0.0;
        ds_imageData[ty][tx][CHANNEL_B] = 0.0;
    }

    __syncthreads();

    float output_r = 0.0;
    float output_g = 0.0;
    float output_b = 0.0;

    if (tx < O_TILE_WIDTH && ty < O_TILE_WIDTH) {
        for (int i = 0; i < Mask_width; i++) {
            for (int j = 0; j < Mask_width; j++) {
                output_r += ds_imageData[ty+i][tx+j][CHANNEL_R] * Mask[i*Mask_width+j];
                output_g += ds_imageData[ty+i][tx+j][CHANNEL_G] * Mask[i*Mask_width+j];
                output_b += ds_imageData[ty+i][tx+j][CHANNEL_B] * Mask[i*Mask_width+j];
            }
        }
    }

    // __syncthreads();

    if (row_o < imageHeight && col_o < imageWidth &&
        tx < O_TILE_WIDTH && ty < O_TILE_WIDTH) {
        outputImageData[(row_o*imageWidth + col_o)*NUM_CHANNELS + CHANNEL_R] = clamp(output_r, 0.0, 1.0);
        outputImageData[(row_o*imageWidth + col_o)*NUM_CHANNELS + CHANNEL_G] = clamp(output_g, 0.0, 1.0);
        outputImageData[(row_o*imageWidth + col_o)*NUM_CHANNELS + CHANNEL_B] = clamp(output_b, 0.0, 1.0);
    }
}


int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE

    assert (imageChannels == NUM_CHANNELS);

    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid((imageWidth-1)/O_TILE_WIDTH+1,
                 (imageHeight-1)/O_TILE_WIDTH+1, 1);

    imageConvolution<<<dimGrid, dimBlock>>>(deviceInputImageData,
                                            deviceOutputImageData,
                                            imageWidth, imageHeight,
                                            deviceMaskData);

    cudaDeviceSynchronize();

    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

