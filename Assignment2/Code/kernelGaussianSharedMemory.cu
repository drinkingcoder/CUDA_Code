//
// Created by drinkingcoder on 17-10-31.
//

#include "kernelGaussianSharedMemory.hpp"

__device__ int MapThreadToGlobal(int tx,int ty) {
    return (blockIdx.x * blockDim.x + tx) + (blockIdx.y * blockDim.y + ty)*IMAGESIZE_WIDTH;
}

__global__ static void kernel(unsigned char *result, unsigned char *ptr, double *gaussian_kernel) {
    __shared__ unsigned char patch[PATCHSIZE_HEIGHT*PATCHSIZE_WIDTH*3];
    __shared__ double cachedGaussian[KERNEL_SIZE*KERNEL_SIZE];

    int x = blockIdx.x*blockDim.x+threadIdx.x;
    x = x - blockIdx.x*KERNEL_SIZE;
    if( x >= IMAGESIZE_WIDTH ) return;

    int y = blockIdx.y*blockDim.y+threadIdx.y;
    y = y - blockIdx.y*KERNEL_SIZE;
    if( y >= IMAGESIZE_HEIGHT ) return;
    int offset = (x+y*IMAGESIZE_WIDTH)*3;

    int px = threadIdx.x, py = threadIdx.y;
    int poffset =  ( px + py*blockDim.x);
    if( poffset < KERNEL_SIZE*KERNEL_SIZE )
        cachedGaussian[poffset] = gaussian_kernel[poffset];
    poffset*=3;

    patch[poffset] = ptr[offset];
    patch[poffset+1] = ptr[offset+1];
    patch[poffset+2] = ptr[offset+2];

    if( px >= blockDim.x - KERNEL_SIZE ) return;
    if( py >= blockDim.y - KERNEL_SIZE ) return;
    __syncthreads();

    for(int channel = 0; channel<3; channel++)
    {
        double tmp_result = 0;
        for( int i = 0; i < KERNEL_SIZE; i++)
            for( int j = 0; j < KERNEL_SIZE; j++) {
//                if ((poffset + i * IMAGESIZE_WIDTH + j) * 3 + channel >= IMAGESIZE_WIDTH * IMAGESIZE_HEIGHT * 3)
//                    printf("!!!!!");
                tmp_result += patch[poffset + (i * PATCHSIZE_WIDTH + j) * 3 + channel] * cachedGaussian[i * KERNEL_SIZE + j];
//                tmp_result += ptr[channel + (offset + i * IMAGESIZE_WIDTH + j)*3] * gaussian_kernel[i * KERNEL_SIZE + j];
            }
        result[offset + channel] = (unsigned char) (tmp_result);
    }
}

__global__ static void kernel2(unsigned char *result, unsigned char *ptr, double *gaussian_kernel) {
    __shared__ unsigned char patch[(PATCHSIZE_WIDTH+KERNEL_SIZE-1)*(PATCHSIZE_HEIGHT+KERNEL_SIZE-1)*3];
    __shared__ double cachedGaussian[KERNEL_SIZE*KERNEL_SIZE];

    int x = blockIdx.x*blockDim.x+threadIdx.x;

    int y = blockIdx.y*blockDim.y+threadIdx.y;

    int offset;

    if( threadIdx.x < KERNEL_SIZE && threadIdx.y < KERNEL_SIZE ) {
        int poffset =  ( threadIdx.x + threadIdx.y*KERNEL_SIZE );
        cachedGaussian[poffset] = gaussian_kernel[poffset];
    }

    int px,py,poffset;
    int patchSizeX = KERNEL_SIZE-1 + blockDim.x;

    py = threadIdx.y + KERNEL_SIZE/2;
    px = threadIdx.x + KERNEL_SIZE/2;
    poffset = (py*patchSizeX + px)*3;
    offset = (x+y*IMAGESIZE_WIDTH)*3;

    patch[poffset] = ptr[offset];
    patch[poffset+1] = ptr[offset+1];
    patch[poffset+2] = ptr[offset+2];

    if( x < KERNEL_SIZE/2 || x >= IMAGESIZE_WIDTH - KERNEL_SIZE/2 ) return;
    if( y < KERNEL_SIZE/2 || y >= IMAGESIZE_HEIGHT - KERNEL_SIZE/2 ) return;

    if( threadIdx.x < KERNEL_SIZE/2 )
    {
        px = threadIdx.x;
        py = KERNEL_SIZE/2 + threadIdx.y;
        poffset = (py*patchSizeX + px)*3;
        offset = (x - KERNEL_SIZE/2 + y*IMAGESIZE_WIDTH)*3;
        patch[poffset + 0]  = ptr[ offset + 0 ];
        patch[poffset + 1]  = ptr[ offset + 1 ];
        patch[poffset + 2]  = ptr[ offset + 2 ];
    }
    if( threadIdx.x >= blockDim.x - KERNEL_SIZE/2 )
    {
        px = KERNEL_SIZE-1 + threadIdx.x;
        py = KERNEL_SIZE/2 + threadIdx.y;
        poffset = (py*patchSizeX + px)*3;
        offset = (x + KERNEL_SIZE/2 + y*IMAGESIZE_WIDTH)*3;
        patch[poffset + 0]  = ptr[ offset + 0 ];
        patch[poffset + 1]  = ptr[ offset + 1 ];
        patch[poffset + 2]  = ptr[ offset + 2 ];
    }
    if( threadIdx.y < KERNEL_SIZE/2 )
    {
        px = KERNEL_SIZE/2 + threadIdx.x;
        py = threadIdx.y;
        poffset = (py*patchSizeX + px)*3;
        offset = (x + (y-KERNEL_SIZE/2)*IMAGESIZE_WIDTH)*3;
        patch[poffset + 0]  = ptr[ offset + 0 ];
        patch[poffset + 1]  = ptr[ offset + 1 ];
        patch[poffset + 2]  = ptr[ offset + 2 ];
    }
    if( threadIdx.y >= blockDim.y - KERNEL_SIZE/2 )
    {
        px = KERNEL_SIZE/2 + threadIdx.x;
        py = KERNEL_SIZE-1 + threadIdx.y;
        poffset = (py*patchSizeX + px)*3;
        offset = (x + (y+KERNEL_SIZE/2)*IMAGESIZE_WIDTH)*3;
        patch[poffset + 0]  = ptr[ offset + 0 ];
        patch[poffset + 1]  = ptr[ offset + 1 ];
        patch[poffset + 2]  = ptr[ offset + 2 ];
    }
    if( threadIdx.x < KERNEL_SIZE/2 && threadIdx.y < KERNEL_SIZE/2 )
    {
        px = threadIdx.x;
        py = threadIdx.y;
        poffset = (py*patchSizeX + px)*3;
        offset = (x - KERNEL_SIZE/2 + (y - KERNEL_SIZE/2)*IMAGESIZE_WIDTH)*3;
        patch[poffset + 0]  = ptr[ offset + 0 ];
        patch[poffset + 1]  = ptr[ offset + 1 ];
        patch[poffset + 2]  = ptr[ offset + 2 ];
    }
    if( threadIdx.x < KERNEL_SIZE/2 && threadIdx.y >= blockDim.y - KERNEL_SIZE/2 )
    {
        px = threadIdx.x;
        py = KERNEL_SIZE-1 + threadIdx.y;
        poffset = (py*patchSizeX + px)*3;
        offset = (x - KERNEL_SIZE/2 + (y + KERNEL_SIZE/2)*IMAGESIZE_WIDTH)*3;
        patch[poffset + 0]  = ptr[ offset + 0 ];
        patch[poffset + 1]  = ptr[ offset + 1 ];
        patch[poffset + 2]  = ptr[ offset + 2 ];
    }
    if( threadIdx.x >= blockDim.x - KERNEL_SIZE/2 && threadIdx.y >= blockDim.y - KERNEL_SIZE/2 )
    {
        px = KERNEL_SIZE-1 + threadIdx.x;
        py = KERNEL_SIZE-1 + threadIdx.y;
        poffset = (py*patchSizeX + px)*3;
        offset = (x + KERNEL_SIZE/2 + (y + KERNEL_SIZE/2)*IMAGESIZE_WIDTH)*3;
        patch[poffset + 0]  = ptr[ offset + 0 ];
        patch[poffset + 1]  = ptr[ offset + 1 ];
        patch[poffset + 2]  = ptr[ offset + 2 ];
    }
    if( threadIdx.x >= KERNEL_SIZE/2 && threadIdx.y < KERNEL_SIZE/2 )
    {
        px = KERNEL_SIZE-1 + threadIdx.x;
        py = threadIdx.y;
        poffset = (py*patchSizeX + px)*3;
        offset = (x + KERNEL_SIZE/2 + (y - KERNEL_SIZE/2)*IMAGESIZE_WIDTH)*3;
        patch[poffset + 0]  = ptr[ offset + 0 ];
        patch[poffset + 1]  = ptr[ offset + 1 ];
        patch[poffset + 2]  = ptr[ offset + 2 ];
    }

    __syncthreads();

    offset = (x+y*IMAGESIZE_WIDTH)*3;
    poffset = (threadIdx.x + threadIdx.y*patchSizeX)*3;
    for(int channel = 0; channel<3; channel++)
    {
        double tmp_result = 0;
        for( int i = 0; i < KERNEL_SIZE; i++)
            for( int j = 0; j < KERNEL_SIZE; j++) {
                tmp_result += patch[poffset + (i * patchSizeX + j) * 3 + channel] * cachedGaussian[i * KERNEL_SIZE + j];
            }
        result[offset + channel] = (unsigned char) (tmp_result);
    }
}

__global__ static void kernel3(unsigned char *result, unsigned char *ptr, double *gaussian_kernel) {
    __shared__ unsigned char patch[(PATCHSIZE_WIDTH+KERNEL_SIZE-1)*(PATCHSIZE_HEIGHT+KERNEL_SIZE-1)*3];
    __shared__ double cachedGaussian[KERNEL_SIZE*KERNEL_SIZE];

    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    int offset;

//    if( threadIdx.x < KERNEL_SIZE && threadIdx.y < KERNEL_SIZE ) {
//        int poffset =  ( threadIdx.x + threadIdx.y*KERNEL_SIZE );
//        cachedGaussian[poffset] = gaussian_kernel[poffset];
//    }

    int px,py,poffset,tx = x,ty = y;
    int patchSizeX = KERNEL_SIZE-1 + blockDim.x;
    bool flag = false;

    if( x < KERNEL_SIZE/2 || x >= IMAGESIZE_WIDTH - KERNEL_SIZE/2 ) return;
    if( y < KERNEL_SIZE/2 || y >= IMAGESIZE_HEIGHT - KERNEL_SIZE/2 ) return;

//    if( threadIdx.y < KERNEL_SIZE/2 ) {
        flag = true;
        tx = threadIdx.x;
        ty = threadIdx.y - KERNEL_SIZE/2;
//    } else if(threadIdx.y < KERNEL_SIZE/2 * 2 ) {
//        flag = true;
//        tx = threadIdx.y - KERNEL_SIZE/2 * 2;
//        ty = threadIdx.x;
//    } else if(threadIdx.y < KERNEL_SIZE/2 * 3) {
//        flag = true;
//        tx = threadIdx.y - KERNEL_SIZE/2 * 2 + blockDim.x;
//        ty = threadIdx.x;
//    } else if(threadIdx.y < KERNEL_SIZE/2 * 4){
//        flag = true;
//        tx = threadIdx.x;
//        ty = threadIdx.y - KERNEL_SIZE/2 * 3 + blockDim.y;
//    }
//    if(flag) {
//        px = tx + KERNEL_SIZE/2;
//        py = ty + KERNEL_SIZE/2;
//        poffset = (py*patchSizeX + px)*3;
//        offset = MapThreadToGlobal(tx, ty)*3;
//        patch[poffset + 0]  = ptr[ offset + 0 ];
//        patch[poffset + 1]  = ptr[ offset + 1 ];
//        patch[poffset + 2]  = ptr[ offset + 2 ];
////        flag = false;
//    }
//    if(KERNEL_SIZE/2 * 4 <= threadIdx.y && threadIdx.y < KERNEL_SIZE/2 * 5) {
//        if(threadIdx.x < KERNEL_SIZE/2) {
//            flag = true;
//            tx = threadIdx.x - KERNEL_SIZE/2;
//            ty = threadIdx.y - KERNEL_SIZE/2 * 5;
//        } else if(threadIdx.x < KERNEL_SIZE/2 * 2) {
//            flag = true;
//            tx = threadIdx.x - KERNEL_SIZE/2 * 2;
//            ty = threadIdx.y - KERNEL_SIZE/2 * 4 + blockDim.y;
//        } else if(threadIdx.x < KERNEL_SIZE/2 * 3) {
//            flag = true;
//            tx = threadIdx.x - KERNEL_SIZE/2 * 2 + blockDim.x;
//            ty = threadIdx.y - KERNEL_SIZE/2 * 4 + blockDim.y;
//        } else if(threadIdx.x < KERNEL_SIZE/2 * 4) {
//            flag = true;
//            tx = threadIdx.x - KERNEL_SIZE/2 * 3 + blockDim.x;
//            ty = threadIdx.y - KERNEL_SIZE/2 * 5;
//        }
//    }
//    if(flag) {
//        px = tx + KERNEL_SIZE/2;
//        py = ty + KERNEL_SIZE/2;
//        poffset = (py*patchSizeX + px)*3;
//        offset = MapThreadToGlobal(tx, ty)*3;
//        patch[poffset + 0]  = ptr[ offset + 0 ];
//        patch[poffset + 1]  = ptr[ offset + 1 ];
//        patch[poffset + 2]  = ptr[ offset + 2 ];
//    }
//    py = threadIdx.y + KERNEL_SIZE/2;
//    px = threadIdx.x + KERNEL_SIZE/2;
//    poffset = (py*patchSizeX + px)*3;
//    offset = (x+y*IMAGESIZE_WIDTH)*3;
//
//    patch[poffset] = ptr[offset];
//    patch[poffset+1] = ptr[offset+1];
//    patch[poffset+2] = ptr[offset+2];
//
//    __syncthreads();

    offset = (x+y*IMAGESIZE_WIDTH);
    poffset = (threadIdx.x + threadIdx.y*patchSizeX);
    for(int channel = 0; channel<3; channel++)
    {
        double tmp_result = 0;
        for( int i = 0; i < KERNEL_SIZE; i++)
            for( int j = 0; j < KERNEL_SIZE; j++) {
                tmp_result += patch[(poffset + i * patchSizeX + j) * 3 + channel] * cachedGaussian[i * KERNEL_SIZE + j];
            }
        result[offset*3 + channel] = (unsigned char) (tmp_result);
    }
}
void KernelGaussianSharedMemory::awakeKernel(){
#define BLOCKSIZE 32
#define GRIDSIZEX (IMAGESIZE_WIDTH/BLOCKSIZE+1)
#define GRIDSIZEY (IMAGESIZE_HEIGHT/BLOCKSIZE+1)
    dim3 gridDim(GRIDSIZEX,GRIDSIZEY);
    dim3 blockDim(BLOCKSIZE,BLOCKSIZE);

    kernel3<<<gridDim,blockDim>>>(m_devResultBitmap,m_devInputBitmap,m_devGaussianKernel);
}