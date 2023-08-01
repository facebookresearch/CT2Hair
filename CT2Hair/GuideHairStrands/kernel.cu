// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <float.h>
#include <math_constants.h>

#define CUB_STDERR
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>

#include "cuda_types.h"
#include "clock.h"

#define BLOCK_SIZE_MEANSHIFT 128    // 32, 64, 128, 256, 512, 1024
#define MAX_ITEMS_PER_THREAD 640
#define HAIR_USE_FAST_CUDA_MATH true

#define CUB_REDUCTION_ALGORITHM cub::BLOCK_REDUCE_WARP_REDUCTIONS

__inline__ __device__ float Gaussian_cu(float x, float sigma)
{
#if HAIR_USE_FAST_CUDA_MATH
    return __expf(-(x * x) / (2 * sigma * sigma));
#else
    return expf(-(x * x) / (2 * sigma * sigma));
#endif
}

__inline__ __device__ void m_assert(bool expression, int line = 0)
{
    if (!expression)
        printf("Assert fail: %s (%d)\n", __FILE__, line);
}

__inline__ __device__ float rad2deg(float rad)
{
#if HAIR_USE_FAST_CUDA_MATH
    return rad * 57.2957795131f;
#else
    return rad / CUDART_PI_F * 180.f;
#endif
}

__inline__ __device__ float deg2rad(float deg) {
#if HAIR_USE_FAST_CUDA_MATH
    return deg * 0.01745329252f;
#else
    return deg / 180.f * CUDART_PI_F;
#endif
}

__inline__ __device__ Point3D operator*(float w, Point3D pin)
{
    Point3D p = pin;
    p.pos = w * p.pos;
    p.dir = w * p.dir;
    return p;
}

__inline__ __device__ Point3D operator+(Point3D a, Point3D b)
{
    Point3D c;
    c.pos = a.pos + b.pos;
    c.dir = a.dir + b.dir;
    return c;
}

__inline__ __device__ void operator+=(Point3D& a, Point3D b)
{
    a.pos += b.pos;
    a.dir += b.dir;
}

__inline__ __device__ float distance_cu(Point3D p1, Point3D p2)
{
#if HAIR_USE_FAST_CUDA_MATH
    float3 v = p1.pos - p2.pos;
    return __fsqrt_rn(dot(v, v));
#else
    return length(p1.pos - p2.pos);
#endif
}

__inline__ __device__ float length_cu(float3 v)
{
#if HAIR_USE_FAST_CUDA_MATH
    return __fsqrt_rn(dot(v, v));
#else
    return length(v);
#endif
}

__inline__ __device__ float3 normalize_cu(float3 v)
{
#if HAIR_USE_FAST_CUDA_MATH
    float invLen = __frsqrt_rn(dot(v, v));
    return v * invLen;
#else
    return normalize(v);
#endif
}

__inline__ __device__ int getIndex_cu(int3 grid_res, int idxx, int idxy, int idxz)
{
    int idx = idxz * grid_res.x * grid_res.y + idxy * grid_res.x + idxx;
    return idx;
}

__inline__ __device__ void getNeighborVoxels_cu(
    int3 grid_res,
    int grid_idx,
    int* vec_grid_idx)
{

    int idxx = grid_idx % grid_res.x;
    int idxy = (grid_idx % (grid_res.x * grid_res.y)) / grid_res.x;
    int idxz = grid_idx / (grid_res.x * grid_res.y);

    if (idxx >= 0 && idxx < grid_res.x
        && idxy >= 0 && idxy < grid_res.y
        && idxz >= 0 && idxz < grid_res.z)
    {}
    else
        printf("Wrong index\n");

    int idx = getIndex_cu(grid_res, idxx, idxy, idxz);
    m_assert(idx == grid_idx, __LINE__);

    int cnt = 0;

    // idxx
    if (idxy + 1 < grid_res.y)
    {
        vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx, idxy + 1, idxz);
        if (idxz + 1 < grid_res.z)
            vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx, idxy + 1, idxz + 1);
        if (idxz - 1 >= 0)
            vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx, idxy + 1, idxz - 1);
    }

    if (idxy - 1 >= 0)
    {
        vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx, idxy - 1, idxz);
        if (idxz + 1 < grid_res.z)
            vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx, idxy - 1, idxz + 1);
        if (idxz - 1 >= 0)
            vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx, idxy - 1, idxz - 1);
    }

    vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx, idxy, idxz);
    if (idxz + 1 < grid_res.z)
        vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx, idxy, idxz + 1);
    if (idxz - 1 >= 0)
        vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx, idxy, idxz - 1);

    // idxx + 1
    if (idxx + 1 < grid_res.x)
    {
        if (idxy + 1 < grid_res.y)
        {
            vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx + 1, idxy + 1, idxz);
            if (idxz + 1 < grid_res.z)
                vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx + 1, idxy + 1, idxz + 1);
            if (idxz - 1 >= 0)
                vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx + 1, idxy + 1, idxz - 1);
        }

        if (idxy - 1 >= 0)
        {
            vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx + 1, idxy - 1, idxz);
            if (idxz + 1 < grid_res.z)
                vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx + 1, idxy - 1, idxz + 1);
            if (idxz - 1 >= 0)
                vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx + 1, idxy - 1, idxz - 1);
        }

        vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx + 1, idxy, idxz);
        if (idxz + 1 < grid_res.z)
            vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx + 1, idxy, idxz + 1);
        if (idxz - 1 >= 0)
            vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx + 1, idxy, idxz - 1);
    }

    if (idxx - 1 >= 0)
    {
        if (idxy + 1 < grid_res.y)
        {
            vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx - 1, idxy + 1, idxz);
            if (idxz + 1 < grid_res.z)
                vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx - 1, idxy + 1, idxz + 1);
            if (idxz - 1 >= 0)
                vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx - 1, idxy + 1, idxz - 1);
        }

        if (idxy - 1 >= 0)
        {
            vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx - 1, idxy - 1, idxz);
            if (idxz + 1 < grid_res.z)
                vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx - 1, idxy - 1, idxz + 1);
            if (idxz - 1 >= 0)
                vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx - 1, idxy - 1, idxz - 1);
        }

        vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx - 1, idxy, idxz);
        if (idxz + 1 < grid_res.z)
            vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx - 1, idxy, idxz + 1);
        if (idxz - 1 >= 0)
            vec_grid_idx[cnt++] = getIndex_cu(grid_res, idxx - 1, idxy, idxz - 1);
    }

    while (cnt < 27)
        vec_grid_idx[cnt++] = -1;
    m_assert(cnt == 27);
}

__inline__ __device__ Point3D linePlaneIntersect_cu(Point3D line, Point3D pt)
{
    Point3D intersect_pt;
    intersect_pt.pos = make_float3(0.f, 0.f, 0.f);
    intersect_pt.dir = make_float3(0.f, 0.f, 0.f);

    float eps = 1e-16;

    float3 V0 = pt.pos;
    float3 n = pt.dir;

    float3 P0 = line.pos;
    float3 u = line.dir;

    float nu = dot(n, u);
    if (fabsf(nu) < eps)
        return intersect_pt;

    float3 w = P0 - V0;
    float s1 = dot(n, -w) / nu;

    intersect_pt.pos = V0 + w + s1 * u;
    intersect_pt.dir = u;

    return intersect_pt;
}

__inline__ __device__ float orientDifference_cu(Point3D p1, Point3D p2)
{
    float3 n1 = normalize_cu(p1.dir);
    float3 n2 = normalize_cu(p2.dir);

    float absdot = fabsf(dot(n1, n2));
    absdot = __saturatef(absdot);

    float rad = acosf(absdot);
    float deg = rad2deg(rad);

    return deg;
}

__inline__ __device__ void getNeighborPoints_cu(
    Point3D* d_incloud,
    Point3D pt,
    int2* d_grid_idx,
    int* d_grid_data,
    int* vec_grid_idx,
    float nei_radius,
    int B,
    int tid,
    Point3D* neipts,
    int* items_per_thread)
{
    int numnei_total = 0;
    int idx = tid;

    for (int i = 0; i < 27; ++i)
    {
        int neiidx = vec_grid_idx[i];
        if (neiidx < 0)
            continue;

        int dataidx = d_grid_idx[neiidx].x;
        int length = d_grid_idx[neiidx].y;

        for (int j = dataidx; j < dataidx + length; ++j)
        {
            if ((*items_per_thread) > MAX_ITEMS_PER_THREAD - 1)
                return;

            int ptidx = d_grid_data[j];
            Point3D ptnei = d_incloud[ptidx];
            float dist = distance_cu(pt, ptnei);
            if (dist < nei_radius)
            {
                if (numnei_total == idx)
                {
                    // flip directions
                    if (dot(pt.dir, ptnei.dir) < 0.f)
                        ptnei.dir *= -1.f;
                    neipts[(*items_per_thread)++] = ptnei;
                    idx += B;
                }
                numnei_total++;
            }
        }
    }
}

__global__ void k_meanShiftCUDA(
    Point3D* d_incloud,
    Point3D* d_outcloud,
    int2* d_grid_idx,
    int* d_grid_data,
    int3 grid_res,
    int numvoxels,
    int numpts,
    float nei_radius,
    int nei_thres,
    float sigma_e,
    float sigma_o,
    float thres_shift,
    float max_num_shift,
    volatile int *progress)
{
    const int pid = blockIdx.x;
    const int tid = threadIdx.x;
    const int B = blockDim.x;
    __shared__ int numnei;
    __shared__ float shift;
    __shared__ int count;
    __shared__ int grid_idx;
    __shared__ Point3D pt;
    __shared__ Point3D pt_old;
    __shared__ Point3D pt_new;
    __shared__ int vec_grid_idx[27]; // 3 x 3 x 3 neighbor voxels

    Point3D ptsum_t;
    Point3D neipts[MAX_ITEMS_PER_THREAD]; // neighboring points (partitioned for each thread)
    int items_per_thread = 0;

    // initialize
    Point3D ptzero;
    ptzero.pos = make_float3(0.f, 0.f, 0.f);
    ptzero.dir = make_float3(0.f, 0.f, 0.f);
    for (int i = 0; i < MAX_ITEMS_PER_THREAD; ++i)
        neipts[i] = ptzero;

    if (tid == 0)
    {
        pt = d_incloud[pid];
        grid_idx = pt.grid_idx;
        getNeighborVoxels_cu(grid_res, grid_idx, vec_grid_idx);
        pt_old = pt;
        shift = 9999999.9f;
        count = 0;
    }

    __syncthreads();

    getNeighborPoints_cu(d_incloud, pt, d_grid_idx, d_grid_data, vec_grid_idx, nei_radius, B, tid, neipts, &items_per_thread);

    __syncthreads();

    typedef cub::BlockReduce<int, BLOCK_SIZE_MEANSHIFT, CUB_REDUCTION_ALGORITHM> BlockReduceIntT;
    __shared__ typename BlockReduceIntT::TempStorage temp_storage_int;
    int items_total = BlockReduceIntT(temp_storage_int).Sum(items_per_thread);

    // get number of neighbors
    if (tid == 0)
    {
        numnei = items_total;
    }
    __syncthreads();

    // simple noise removal
    if (numnei < nei_thres)
    {
        if (tid == 0)
            d_outcloud[pid] = ptzero;
        return;
    }

    while (shift > thres_shift && count < max_num_shift)
    {
        float wsum_t = 0.f;
        ptsum_t.pos = make_float3(0.f, 0.f, 0.f);
        ptsum_t.dir = make_float3(0.f, 0.f, 0.f);

        if (tid < numnei)
        {
            for (int i = 0; i < items_per_thread; ++i)
            {
                Point3D ptinter = linePlaneIntersect_cu(neipts[i], pt_old);
                float dist_e = distance_cu(pt_old, ptinter);
                float diff_o = orientDifference_cu(pt_old, ptinter);
                float w_e = Gaussian_cu(dist_e, sigma_e);
                float w_o = Gaussian_cu(diff_o, sigma_o);
                float w = w_e * w_o;
                wsum_t += w;
                ptsum_t += (w * ptinter);
            }
        }
        __syncthreads();

        typedef cub::BlockReduce<Point3D, BLOCK_SIZE_MEANSHIFT, CUB_REDUCTION_ALGORITHM> BlockReduceT;
        __shared__ typename BlockReduceT::TempStorage temp_storage;
        Point3D ptsum = BlockReduceT(temp_storage).Sum(ptsum_t);

        typedef cub::BlockReduce<float, BLOCK_SIZE_MEANSHIFT, CUB_REDUCTION_ALGORITHM> BlockReduceFloatT;
        __shared__ typename BlockReduceFloatT::TempStorage temp_storage_float;
        float wsum = BlockReduceFloatT(temp_storage_float).Sum(wsum_t);

        if (tid == 0)
        {
            ptsum.pos /= wsum;
            ptsum.dir /= wsum;
            ptsum.dir = normalize_cu(ptsum.dir);
            pt_new = ptsum;

            shift = distance_cu(pt_old, pt_new);
            pt_old = pt_new;
            count++;
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_outcloud[pid] = pt_new;
    }

    if (tid == 0)
    {
        atomicAdd((int *)progress, 1);
        __threadfence_system();
    }
}

extern "C" void launchMeanShiftCUDA(
    Point3D * d_incloud,
    Point3D * d_outcloud,
    int2 * d_grid_idx,
    int* d_grid_data,
    int3 grid_res,
    int numvoxels,
    int numpts,
    float nei_radius,
    int nei_thres,
    float sigma_e,
    float sigma_o,
    float thres_shift,
    float max_num_shift)
{

    volatile int *device_bid, *bid;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc((void **)&bid, sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer((int **)&device_bid, (int *)bid, 0);

    *bid = 0;
    const dim3 gridSize(numpts, 1, 1);
    const dim3 blockSize(BLOCK_SIZE_MEANSHIFT, 1, 1);

    printf("Running MeanShift with CUDA...\n");

    k_meanShiftCUDA <<<gridSize, blockSize >>> (
        d_incloud,
        d_outcloud,
        d_grid_idx,
        d_grid_data,
        grid_res,
        numvoxels,
        numpts,
        nei_radius,
        nei_thres,
        sigma_e,
        sigma_o,
        thres_shift,
        max_num_shift,
        device_bid);
    
    Clock clk;
    clk.tick();
    
    unsigned int numBlocks = gridSize.x * gridSize.y * gridSize.z;
    float currentProgress = 0.0f;
    do
    {
        if (currentProgress == 0.0f)
            if(clk.tock() > 120)
                break;  // to avoid forver loop with kernel launch error

        int currentBlock = *bid;
        float kernProgress = (float)currentBlock / (float)numBlocks;
        if ((kernProgress - currentProgress) > 0.001f)
        {
          printf("\rMeanShifting: %2.1f%%", (kernProgress * 100));
          currentProgress = kernProgress;
        }
    } while (currentProgress < 0.999f);

    if (currentProgress > 0)
        printf("\rMeanShifting: 100.0%%\n");
    else
        printf("\rMeanShifting: 0.0%%\n");
}