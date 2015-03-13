// Excercise 6 - Histogram Equalization

/******************************************************************************
 *                                                                            *
 * FIXME: nefunguje computeCDF, zatim se resi na hostu                        *
 *                                                                            *
 ******************************************************************************/

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here

#define BLOCK_WIDTH 16

#define NUM_CHANNELS 3
#define CHANNEL_R 0
#define CHANNEL_G 1
#define CHANNEL_B 2

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void castImageData(float * inputImageData,
                              unsigned char * imageBuffer,
                              int imageWidth, int imageHeight) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < imageHeight && col < imageWidth) {
        imageBuffer[(row*imageWidth + col) * NUM_CHANNELS + CHANNEL_R]
            = (unsigned char) (255 * inputImageData[(row*imageWidth + col) * NUM_CHANNELS + CHANNEL_R]);
        imageBuffer[(row*imageWidth + col) * NUM_CHANNELS + CHANNEL_G]
            = (unsigned char) (255 * inputImageData[(row*imageWidth + col) * NUM_CHANNELS + CHANNEL_G]);
        imageBuffer[(row*imageWidth + col) * NUM_CHANNELS + CHANNEL_B]
            = (unsigned char) (255 * inputImageData[(row*imageWidth + col) * NUM_CHANNELS + CHANNEL_B]);
    }
}

__global__ void convertImageToGrayscale(unsigned char * imageBuffer,
                                        unsigned char * grayscaleImageData,
                                        int imageWidth, int imageHeight) {
    unsigned char r, g, b;

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < imageHeight && col < imageWidth) {
        r = imageBuffer[(row*imageWidth + col) * NUM_CHANNELS + CHANNEL_R];
        g = imageBuffer[(row*imageWidth + col) * NUM_CHANNELS + CHANNEL_G];
        b = imageBuffer[(row*imageWidth + col) * NUM_CHANNELS + CHANNEL_B];

        grayscaleImageData[row*imageWidth + col] = (unsigned char) (0.21f*r + 0.71f*g + 0.07f*b);
    }
}

__global__ void computeImageHistogram(unsigned char * grayscaleImageData,
                                      unsigned int * histogram,
                                      int length) {

    __shared__ unsigned int histogramPrivate[HISTOGRAM_LENGTH];

    int tx = threadIdx.x;

    if (tx < HISTOGRAM_LENGTH)
        histogramPrivate[tx] = 0;
    __syncthreads();

    int i = blockIdx.x*blockDim.x + tx;

    if (i < length)
        atomicAdd(&histogramPrivate[grayscaleImageData[i]], 1);
    __syncthreads();

    if (tx < HISTOGRAM_LENGTH) {
        atomicAdd(&histogram[tx], histogramPrivate[tx]);
    }
}

// probability of a pixel to be in a histogram bin
#define p(x, size) ((float) (x) / (size))

__global__ void computeCDF(unsigned int * histogram, float * cdf, int size) {

    __shared__ float XY[HISTOGRAM_LENGTH];

    if (threadIdx.x < HISTOGRAM_LENGTH) {
        if (threadIdx.x == 0)
            XY[threadIdx.x] = p(histogram[threadIdx.x], size);
        else
            XY[threadIdx.x] = histogram[threadIdx.x];

        __syncthreads();

        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int index = (threadIdx.x+1) * stride * 2 - 1;
            if (index < blockDim.x)
                XY[index] += p(XY[index-stride], size);
            __syncthreads();
        }

        for (int stride = blockDim.x/4; stride > 0; stride /= 2) {
            __syncthreads();
            int index = (threadIdx.x+1) * stride * 2 - 1;
            if (index + stride < blockDim.x)
                XY[index+stride] += p(XY[index], size);
        }

        __syncthreads();

        printf("%d: %f\n", threadIdx.x, XY[threadIdx.x]);

        cdf[threadIdx.x] = XY[threadIdx.x];
    }
}

#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))

#define clamp(x, start, end) min(max(x, start), end)

// cdf[0] == cdfmin
#define correct_color(val, cdf) clamp(255*(cdf[val] - cdf[0])/(1 - cdf[0]), 0, 255)

__global__ void correctImage(unsigned char * imageBuffer, float * outputImageData,
                             float * cdf, int imageWidth, int imageHeight) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < imageHeight && col < imageWidth) {
        outputImageData[(row*imageWidth + col) * NUM_CHANNELS + CHANNEL_R]
            = (float) (correct_color(imageBuffer[(row*imageWidth + col) * NUM_CHANNELS + CHANNEL_R], cdf) / 255.0);
        outputImageData[(row*imageWidth + col) * NUM_CHANNELS + CHANNEL_G]
            = (float) (correct_color(imageBuffer[(row*imageWidth + col) * NUM_CHANNELS + CHANNEL_G], cdf) / 255.0);
        outputImageData[(row*imageWidth + col) * NUM_CHANNELS + CHANNEL_B]
            = (float) (correct_color(imageBuffer[(row*imageWidth + col) * NUM_CHANNELS + CHANNEL_B], cdf) / 255.0);
    }
}


int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;

    //@@ Insert more code here

    float * deviceInputImageData;
    float * deviceOutputImageData;
    unsigned char * deviceImageBuffer;
    unsigned char * deviceGrayscaleImageData;
    unsigned int * deviceHistogram;
    float * deviceCDF;

    //##

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here

    assert (imageChannels == NUM_CHANNELS);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");

    wbCheck(cudaMalloc((void **) &deviceInputImageData,
                       imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceImageBuffer,
                       imageWidth * imageHeight * imageChannels * sizeof(unsigned char)));
    wbCheck(cudaMalloc((void **) &deviceOutputImageData,
                       imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceGrayscaleImageData,
                       imageWidth * imageHeight * sizeof(unsigned char)));
    wbCheck(cudaMalloc((void **) &deviceHistogram,
                       HISTOGRAM_LENGTH * sizeof(unsigned int)));
    wbCheck(cudaMalloc((void **) &deviceCDF,
                       HISTOGRAM_LENGTH * sizeof(float)));

    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(GPU, "Clearing histogram memory.");
    wbCheck(cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int)));
    wbTime_stop(GPU, "Clearing histogram memory.");

    wbTime_start(Copy, "Copying data to the GPU");
    wbCheck(cudaMemcpy(deviceInputImageData,
                       hostInputImageData,
                       imageWidth * imageHeight * imageChannels * sizeof(float),
                       cudaMemcpyHostToDevice));
    wbTime_stop(Copy, "Copying data to the GPU");

    dim3 _2D_dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 _2D_dimGrid((imageWidth-1)/BLOCK_WIDTH+1,
                     (imageHeight-1)/BLOCK_WIDTH+1);

    dim3 _1D_dimBlock(HISTOGRAM_LENGTH);
    dim3 _1D_dimGrid((imageWidth*imageHeight-1)/HISTOGRAM_LENGTH+1);

    wbTime_start(Compute, "Doing the computation on the GPU");

    castImageData<<<_2D_dimGrid, _2D_dimBlock>>>(deviceInputImageData,
                                                 deviceImageBuffer,
                                                 imageWidth, imageHeight);

    cudaDeviceSynchronize();

    convertImageToGrayscale<<<_2D_dimGrid, _2D_dimBlock>>>(deviceImageBuffer,
                                                           deviceGrayscaleImageData,
                                                           imageWidth, imageHeight);

    cudaDeviceSynchronize();

    computeImageHistogram<<<_1D_dimGrid, _1D_dimBlock>>>(deviceGrayscaleImageData,
                                                         deviceHistogram,
                                                         imageWidth*imageHeight);

    cudaDeviceSynchronize();

    // cumulative distribution function
    //computeCDF<<<1, HISTOGRAM_LENGTH>>>(deviceHistogram, deviceCDF,
    //                                    imageWidth*imageHeight);
    //
    

    /*** host-side implementation of CDF ***/

    unsigned int * histogram = (unsigned int *) malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));
    if (histogram == NULL) {
        wbLog(ERROR, "Memory allocation failed - histogram");
        return -1;
    }
    wbCheck(cudaMemcpy(histogram, deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    float * cdf = (float *) malloc(HISTOGRAM_LENGTH * sizeof(float));
    if (cdf == NULL) {
        wbLog(ERROR, "Memory allocation failed - cdf");
        return -1;
    }

    int size = imageWidth * imageHeight;
    cdf[0] = p(histogram[0], size);
    for (int i = 1; i < HISTOGRAM_LENGTH; i++)
        cdf[i] = cdf[i - 1] + p(histogram[i], size);
    
    wbCheck(cudaMemcpy(deviceCDF, cdf, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice));

    /*** host-side implementation of CDF ***/

    correctImage<<<_2D_dimGrid, _2D_dimBlock>>>(deviceImageBuffer,
                                                deviceOutputImageData,
                                                deviceCDF,
                                                imageWidth, imageHeight);

    cudaDeviceSynchronize();

    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
    //##

    wbSolution(args, outputImage);

    //@@ insert code here

    wbCheck(cudaFree(deviceInputImageData));
    wbCheck(cudaFree(deviceOutputImageData));
    wbCheck(cudaFree(deviceImageBuffer));
    wbCheck(cudaFree(deviceGrayscaleImageData));
    wbCheck(cudaFree(deviceHistogram));
    wbCheck(cudaFree(deviceCDF));

    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

