//
// Created by drinkingcoder on 17-10-31.
//

#include "kernelGaussianSharedMemory.hpp"

__global__ static void kernel(unsigned char *result, unsigned char *ptr, double *gaussian_kernel) {
    __shared__ unsigned char patch[PATCHSIZE_HEIGHT*PATCHSIZE_WIDTH*3];
    __shared__ double cachedGaussian[KERNEL_SIZE*KERNEL_SIZE];

//    int x = blockIdx.x*blockDim.x+threadIdx.x;
//    if( x >= IMAGESIZE_WIDTH-KERNEL_SIZE ) return;
//    int y = blockIdx.y*blockDim.y+threadIdx.y;
//    if( y >= IMAGESIZE_HEIGHT-KERNEL_SIZE ) return;
//    int offset = x+y*IMAGESIZE_WIDTH;
//
//    for(int channel = 0;channel<3;channel++) {
//        double tmp_result = 0;
//        for (int i = 0; i < KERNEL_SIZE; i++)
//            for (int j = 0; j < KERNEL_SIZE; j++)
//                tmp_result += ptr[channel + (offset + i * IMAGESIZE_WIDTH + j)*3] * gaussian_kernel[i * KERNEL_SIZE + j];
//        result[offset*3 + channel ] = (unsigned char) (tmp_result);
//    }


    int x = blockIdx.x*blockDim.x+threadIdx.x;
    x = x - (x/PATCHSIZE_WIDTH)*(KERNEL_SIZE/2);
    if( x >= IMAGESIZE_WIDTH ) return;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    if( y >= IMAGESIZE_HEIGHT ) return;
    y = y - (y/PATCHSIZE_HEIGHT)*(KERNEL_SIZE/2);
    int offset = x+y*IMAGESIZE_WIDTH;

    int px = x%PATCHSIZE_WIDTH, py = y%PATCHSIZE_HEIGHT;
//    int nx = x/PATCHSIZE_WIDTH, ny = y/PATCHSIZE_HEIGHT;
    int poffset =  ( px + py*PATCHSIZE_WIDTH );
    if( poffset < KERNEL_SIZE*KERNEL_SIZE ) cachedGaussian[poffset] = gaussian_kernel[poffset];

    patch[poffset*3] = ptr[offset*3];
    patch[poffset*3+1] = ptr[offset*3+1];
    patch[poffset*3+2] = ptr[offset*3+2];

    if( px >= PATCHSIZE_WIDTH - KERNEL_SIZE ) return;
    if( py >= PATCHSIZE_HEIGHT - KERNEL_SIZE ) return;
    __syncthreads();

    for(int channel = 0; channel<3; channel++)
    {
        double tmp_result = 0;
        for( int i = 0; i < KERNEL_SIZE; i++)
            for( int j = 0; j < KERNEL_SIZE; j++) {
//                if ((poffset + i * IMAGESIZE_WIDTH + j) * 3 + channel >= IMAGESIZE_WIDTH * IMAGESIZE_HEIGHT * 3)
//                    printf("!!!!!");
                tmp_result = patch[(poffset + i * IMAGESIZE_WIDTH + j) * 3 + channel] * cachedGaussian[i * KERNEL_SIZE + j];
//                tmp_result += ptr[channel + (offset + i * IMAGESIZE_WIDTH + j)*3] * gaussian_kernel[i * KERNEL_SIZE + j];
            }
        result[offset*3+channel] = (unsigned char) (tmp_result);
    }
}

void KernelGaussianSharedMemory::awakeKernel(){
#define BLOCKSIZE 32
#define GRIDSIZEX (IMAGESIZE_WIDTH/BLOCKSIZE)
#define GRIDSIZEY (IMAGESIZE_HEIGHT/BLOCKSIZE)
    dim3 gridDim(GRIDSIZEX,GRIDSIZEY);
    dim3 blockDim(BLOCKSIZE,BLOCKSIZE);

    kernel<<<gridDim,blockDim>>>(m_devResultBitmap,m_devInputBitmap,m_devGaussianKernel);
}