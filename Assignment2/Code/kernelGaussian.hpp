//
// Created by drinkingcoder on 17-10-31.
//

#ifndef CUDA_GUASSIAN_BLUR_KERNELGAUSSIAN_HPP
#define CUDA_GUASSIAN_BLUR_KERNELGAUSSIAN_HPP

#include "kernel.hpp"

class KernelGaussian : public Kernel {
#define IMAGESIZE_HEIGHT 480
#define IMAGESIZE_WIDTH 640
#define IMAGE_SIZE  (IMAGESIZE_HEIGHT*IMAGESIZE_WIDTH)
#define KERNEL_SIZE     9
#define PATCHSIZE_HEIGHT    24
#define PATCHSIZE_WIDTH     32
public:
    KernelGaussian():Kernel()
            ,m_resultImage(IMAGESIZE_HEIGHT,IMAGESIZE_WIDTH,CV_8UC3) {
    }

    virtual ~KernelGaussian(){}

//    __global__ void kernelRough(unsigned char * result, unsigned char * ptr,double * gaussian_kernel);

    virtual double gaussian_function(double x);
    virtual void compute_gaussian_kernel(double * gaussian_kernel);

    virtual void preparation();
    virtual void awakeKernel();
    virtual void postProcessing();

protected:
    double* m_gaussianKernel;
    double *m_devGaussianKernel;
    unsigned char* m_devInputBitmap;
    unsigned char* m_devResultBitmap;
    cv::Mat m_resultImage;
};

#endif //CUDA_GUASSIAN_BLUR_KERNELGAUSSIAN_HPP
