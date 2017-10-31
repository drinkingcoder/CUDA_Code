/**
  * Maximum number of thread per thread: 1024
  * Warp size: 32
  */


#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

#include "kernelGaussian.hpp"
#include "kernelGaussianSharedMemory.hpp"

using namespace cv;
using namespace std;
//__global__ void kernelSharedMam(unsigned char * result, unsigned char * ptr, double * gaussian_kernel)
//{
//    __shared__ unsigned char patch[PATCHSIZE_HEIGHT*PATCHSIZE_WIDTH*3];
//
//    int x = blockIdx.x*blockDim.x+threadIdx.x;
//    x = x - (x/PATCHSIZE_WIDTH)*(KERNEL_SIZE/2);
//    if( x >= IMAGESIZE_WIDTH ) return;
//    int y = blockIdx.y*blockDim.y+threadIdx.y;
//    if( y >= IMAGESIZE_HEIGHT ) return;
//    y = y - (y/PATCHSIZE_HEIGHT)*(KERNEL_SIZE/2);
//    int offset = x+y*IMAGESIZE_WIDTH;
//
//    int px = x%PATCHSIZE_WIDTH, py = y%PATCHSIZE_HEIGHT;
////    int nx = x/PATCHSIZE_WIDTH, ny = y/PATCHSIZE_HEIGHT;
//    int poffset =  ( px + py*PATCHSIZE_WIDTH ) * 3;
//    patch[poffset] = ptr[offset*3];
//    patch[poffset+1] = ptr[offset*3+1];
//    patch[poffset+2] = ptr[offset*3+2];
//
//    if( px >= PATCHSIZE_WIDTH - KERNEL_SIZE ) return;
//    if( py >= PATCHSIZE_HEIGHT - KERNEL_SIZE ) return;
//    __syncthreads();
//
//    for(int channel = 0; channel<3; channel++)
//    {
//        double tmp_result = 0;
//        for( int i = 0; i < KERNEL_SIZE; i++)
//            for( int j = 0; j < KERNEL_SIZE; j++)
//               tmp_result = patch[poffset + (i * IMAGESIZE_WIDTH + j)*3 + channel] *gaussian_kernel[i*KERNEL_SIZE+j];
//        result[(offset+KERNEL_SIZE/2)*3+channel] = (unsigned char) (tmp_result);
//    }
//}

int main()
{
    KernelTimeTest test;
    std::shared_ptr<Kernel> kernel = make_shared<KernelGaussianSharedMemory>();
    test.testSingleTime(kernel);
//    test.testMultipleTimes(kernel,100);
    std::cout << " duration = " << test.getDuration() << std::endl;
}
