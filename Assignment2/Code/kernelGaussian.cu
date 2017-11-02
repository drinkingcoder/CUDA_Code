//
// Created by drinkingcoder on 17-10-31.
//

#include "kernelGaussian.hpp"
#include <cuda_by_example/book.h>
using namespace cv;

double KernelGaussian::gaussian_function(double x) {
    return 1/sqrt(2*M_PI)*exp(-pow(x,2)/2);
}

void KernelGaussian::compute_gaussian_kernel(double *gaussian_kernel) {
    double sum = 0;
    for(int i=0; i<KERNEL_SIZE; i++)
        for(int j=0; j<KERNEL_SIZE; j++)
        {
            gaussian_kernel[i*KERNEL_SIZE + j] = gaussian_function(fabs(KERNEL_SIZE/2 - i))*gaussian_function(fabs(KERNEL_SIZE/2 - j));
            sum += gaussian_kernel[i*KERNEL_SIZE + j];
        }
    double tot = 0;
    for(int i=0; i<KERNEL_SIZE; i++)
        for(int j=0; j<KERNEL_SIZE; j++) {
            gaussian_kernel[i * KERNEL_SIZE + j] /= sum;
            tot += gaussian_kernel[i * KERNEL_SIZE + j];
        }
}

__global__ static void kernel(unsigned char *result, unsigned char *ptr, double *gaussian_kernel) {
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    if( x >= IMAGESIZE_WIDTH-KERNEL_SIZE ) return;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    if( y >= IMAGESIZE_HEIGHT-KERNEL_SIZE ) return;
    int offset = x+y*IMAGESIZE_WIDTH;

    for(int channel = 0;channel<3;channel++) {
        double tmp_result = 0;
        for (int i = 0; i < KERNEL_SIZE; i++)
            for (int j = 0; j < KERNEL_SIZE; j++)
                tmp_result += ptr[channel + (offset + i * IMAGESIZE_WIDTH + j)*3] * gaussian_kernel[i * KERNEL_SIZE + j];
        result[offset*3 + channel ] = (unsigned char) (tmp_result);
    }
}

void KernelGaussian::preparation() {
    m_gaussianKernel = new double[KERNEL_SIZE*KERNEL_SIZE];
    compute_gaussian_kernel(m_gaussianKernel);

    HANDLE_ERROR(
            cudaMalloc((void**)&m_devGaussianKernel, KERNEL_SIZE*KERNEL_SIZE*sizeof(double))
    );
    HANDLE_ERROR(
            cudaMemcpy(m_devGaussianKernel,m_gaussianKernel,KERNEL_SIZE*KERNEL_SIZE*sizeof(double),cudaMemcpyHostToDevice)
    );

    Mat host_input_image = imread("Image.jpg");
    host_input_image.convertTo(host_input_image,CV_8UC3);
    resize(host_input_image,host_input_image,cv::Size(IMAGESIZE_WIDTH,IMAGESIZE_HEIGHT));

    HANDLE_ERROR(
            cudaMalloc((void**)&m_devInputBitmap, 3*IMAGESIZE_HEIGHT*IMAGESIZE_WIDTH)
    );
    HANDLE_ERROR(
            cudaMemcpy(m_devInputBitmap, host_input_image.data, 3*IMAGESIZE_HEIGHT*IMAGESIZE_WIDTH, cudaMemcpyHostToDevice)
    );

    HANDLE_ERROR(
            cudaMalloc((void**)&m_devResultBitmap, 3*host_input_image.rows*host_input_image.cols)
    );
}

void KernelGaussian::awakeKernel() {
#define BLOCKSIZE 32
#define GRIDSIZEX (IMAGESIZE_WIDTH/BLOCKSIZE+1)
#define GRIDSIZEY (IMAGESIZE_HEIGHT/BLOCKSIZE+1)
    dim3 gridDim(GRIDSIZEX,GRIDSIZEY);
    dim3 blockDim(BLOCKSIZE,BLOCKSIZE);

    kernel<<<gridDim,blockDim>>>(m_devResultBitmap,m_devInputBitmap,m_devGaussianKernel);
}

void KernelGaussian::postProcessing(){
    HANDLE_ERROR(
            cudaMemcpy(m_resultImage.data,m_devResultBitmap,3*IMAGESIZE_WIDTH*IMAGESIZE_HEIGHT, cudaMemcpyDeviceToHost)
    );
//    imwrite("Kernel9x9.jpg",m_resultImage);
    imshow("Image after blurred",m_resultImage);
    waitKey();

    HANDLE_ERROR(
            cudaFree(m_devInputBitmap)
    );
    HANDLE_ERROR(
            cudaFree(m_devResultBitmap)
    );
    HANDLE_ERROR(
            cudaFree(m_devGaussianKernel)
    );
}
