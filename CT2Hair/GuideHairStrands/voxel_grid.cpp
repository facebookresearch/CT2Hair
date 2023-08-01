// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "voxel_grid.h"

VoxelGrid::VoxelGrid(pcl::PointCloud<PointT>::Ptr pcl, float voxel_length)
{
    m_pcl = pcl;
    m_voxel_length = voxel_length;

    initVoxelGrid();
}

VoxelGrid::VoxelGrid(pcl::PointCloud<PointT>::Ptr pcl, int W, int H, int D, float minx, float maxx, float miny, float maxy, float minz, float maxz)
{
    m_pcl = pcl;
    initVoxelGrid(W, H, D, minx, maxx, miny, maxy, minz, maxz);
    computeOrient3D();
}

VoxelGrid::VoxelGrid(pcl::PointCloud<PointT>::Ptr pcl, float voxel_length, float minx, float maxx, float miny, float maxy, float minz, float maxz)
{
    m_pcl = pcl;
    initVoxelGrid(voxel_length, minx, maxx, miny, maxy, minz, maxz);
    computeOrient3D();
}

void VoxelGrid::radiusSearch(const PointT& pt, float radius, std::vector<int>& k_indices)
{
    k_indices.clear();

    std::vector<int> vec_grid_idx = getNeighborVoxels(getVoxelGridIndex(pt));

    for (int grid_idx : vec_grid_idx)
    {
        const std::vector<int>& vec_idx = m_grid[grid_idx];

        for (int idx : vec_idx)
        {
            float dist = pcl::geometry::distance(pt, (*m_pcl)[idx]);
            if (dist < radius)
                k_indices.push_back(idx);
        }
    }

    return;
}

void VoxelGrid::getGridRes(int* m_gridres)
{
    m_gridres[0] = m_grid_res[0];
    m_gridres[1] = m_grid_res[1];
    m_gridres[2] = m_grid_res[2];
}

void VoxelGrid::getGridOrigin(float* origin)
{
    origin[0] = m_origin[0];
    origin[1] = m_origin[1];
    origin[2] = m_origin[2];
}

float VoxelGrid::getGridVoxelLength()
{
    return m_voxel_length;
}

int VoxelGrid::numNonEmptyVoxels()
{
    int cnt = 0;
    for (auto& v : m_grid)
    {
        if (!v.empty())
            cnt++;
    }
    return cnt;
}

Eigen::Vector3f VoxelGrid::getOrientAt(int idx)
{
    if (idx < 0 || idx >= m_grid_res[0] * m_grid_res[1] * m_grid_res[2])
    {
        printf("Invalid grid index\n");
        return Eigen::Vector3f();
    }
    return m_ori[idx];
}

Eigen::Vector3f VoxelGrid::getCenterPositionAt(int idx)
{
    if (idx < 0 || idx >= m_grid_res[0] * m_grid_res[1] * m_grid_res[2])
    {
        printf("Invalid grid index\n");
        return Eigen::Vector3f();
    }

    int idxx = idx % m_grid_res[0];
    int idxy = (idx % (m_grid_res[0] * m_grid_res[1])) / m_grid_res[0];
    int idxz = idx / (m_grid_res[0] * m_grid_res[1]);

    float x = m_origin[0] + idxx * m_voxel_length + m_voxel_length * 0.5f;
    float y = m_origin[1] + idxy * m_voxel_length + m_voxel_length * 0.5f;
    float z = m_origin[2] + idxz * m_voxel_length + m_voxel_length * 0.5f;

    return Eigen::Vector3f(x, y, z);
}

std::vector<int> VoxelGrid::getPointIndicesAt(int idx)
{
    if (idx < 0 || idx >= m_grid_res[0] * m_grid_res[1] * m_grid_res[2])
    {
        printf("Invalid grid index\n");
        return std::vector<int>();
    }

    return m_grid[idx];
}

std::vector<int> VoxelGrid::getPointIndicesAt(int idxx, int idxy, int idxz)
{
    int idx = getIndex(idxx, idxy, idxz);
    if (idx < 0 || idx >= m_grid_res[0] * m_grid_res[1] * m_grid_res[2])
    {
        printf("Invalid grid index\n");
        return std::vector<int>();
    }

    return m_grid[idx];
}

void VoxelGrid::initVoxelGrid()
{
    // bounding box
    float minx = std::numeric_limits<float>::max();
    float miny = std::numeric_limits<float>::max();
    float minz = std::numeric_limits<float>::max();
    float maxx = std::numeric_limits<float>::min();
    float maxy = std::numeric_limits<float>::min();
    float maxz = std::numeric_limits<float>::min();

    for (const auto& pt : *m_pcl)
    {
        if (pt.x < minx) minx = pt.x;
        if (pt.y < miny) miny = pt.y;
        if (pt.z < minz) minz = pt.z;
        if (pt.x > maxx) maxx = pt.x;
        if (pt.y > maxy) maxy = pt.y;
        if (pt.z > maxz) maxz = pt.z;
    }

    // This is for hairFeatureMatchingPair
    // keep it in this way for hairFeatureMatchingPair (maxdistance cross search)
    m_origin[0] = minx - m_voxel_length;
    m_origin[1] = miny - m_voxel_length;
    m_origin[2] = minz - m_voxel_length;
    m_grid_res[0] = std::ceil((maxx + m_voxel_length - m_origin[0]) / m_voxel_length);
    m_grid_res[1] = std::ceil((maxy + m_voxel_length - m_origin[1]) / m_voxel_length);
    m_grid_res[2] = std::ceil((maxz + m_voxel_length - m_origin[2]) / m_voxel_length);

    int W = m_grid_res[0];
    int H = m_grid_res[1];
    int D = m_grid_res[2];

    // voxel grid
    m_grid.resize(m_grid_res[0] * m_grid_res[1] * m_grid_res[2]);
    for (int idx = 0; idx < (*m_pcl).size(); ++idx)
    {
        auto& pt = (*m_pcl)[idx];
        if (pt.x > m_origin[0] && pt.x < m_origin[0] + m_voxel_length * W &&
            pt.y > m_origin[1] && pt.y < m_origin[1] + m_voxel_length * H &&
            pt.z > m_origin[2] && pt.z < m_origin[2] + m_voxel_length * D)
        {
            int grid_idx = getVoxelGridIndex(pt);
            m_grid[grid_idx].push_back(idx);
            pt.label = grid_idx;
        }
    }
}

void VoxelGrid::initVoxelGrid(int W, int H, int D, float minx, float maxx, float miny, float maxy, float minz, float maxz)
{
    float center[3];
    center[0] = (maxx + minx) / 2.f;
    center[1] = (maxy + miny) / 2.f;
    center[2] = (maxz + minz) / 2.f;

    float m_voxel_lengthx = (maxx - minx + 2.f) / float(W);
    float m_voxel_lengthy = (maxy - miny + 2.f) / float(H);
    float m_voxel_lengthz = (maxz - minz + 2.f) / float(D);
    m_voxel_length = std::max(std::max(m_voxel_lengthx, m_voxel_lengthy), m_voxel_lengthz);

    // set origin of the voxel grid
    m_origin[0] = center[0] - m_voxel_length * (float(W) / 2.f);
    m_origin[1] = center[1] - m_voxel_length * (float(H) / 2.f);
    m_origin[2] = center[2] - m_voxel_length * (float(D) / 2.f);

    // get grid resolution
    m_grid_res[0] = W;
    m_grid_res[1] = H;
    m_grid_res[2] = D;

    // voxel grid
    m_grid.resize(m_grid_res[0] * m_grid_res[1] * m_grid_res[2]);
    for (int idx = 0; idx < (*m_pcl).size(); ++idx)
    {
        auto& pt = (*m_pcl)[idx];
        if (pt.x > m_origin[0] && pt.x < m_origin[0] + m_voxel_length * W &&
            pt.y > m_origin[1] && pt.y < m_origin[1] + m_voxel_length * H &&
            pt.z > m_origin[2] && pt.z < m_origin[2] + m_voxel_length * D)
        {
            int grid_idx = getVoxelGridIndex(pt);
            m_grid[grid_idx].push_back(idx);
            pt.label = grid_idx;
        }
    }
}

void VoxelGrid::initVoxelGrid(float voxel_length, float minx, float maxx, float miny, float maxy, float minz, float maxz)
{
    m_voxel_length = voxel_length;

    // set origin of the voxel grid
    m_origin[0] = minx - 0.1f;
    m_origin[1] = miny - 0.1f;
    m_origin[2] = minz - 0.1f;

    // get grid resolution
    m_grid_res[0] = (int)std::ceil((maxx - m_origin[0]) / m_voxel_length + 0.1f);
    m_grid_res[1] = (int)std::ceil((maxy - m_origin[1]) / m_voxel_length + 0.1f);
    m_grid_res[2] = (int)std::ceil((maxz - m_origin[2]) / m_voxel_length + 0.1f);

    int W = m_grid_res[0];
    int H = m_grid_res[1];
    int D = m_grid_res[2];

    // voxel grid
    m_grid.resize(m_grid_res[0] * m_grid_res[1] * m_grid_res[2]);
    for (int idx = 0; idx < (*m_pcl).size(); ++idx)
    {
        auto& pt = (*m_pcl)[idx];
        if (pt.x > m_origin[0] && pt.x < m_origin[0] + m_voxel_length * W &&
            pt.y > m_origin[1] && pt.y < m_origin[1] + m_voxel_length * H &&
            pt.z > m_origin[2] && pt.z < m_origin[2] + m_voxel_length * D)
        {
            int grid_idx = getVoxelGridIndex(pt);
            m_grid[grid_idx].push_back(idx);
            pt.label = grid_idx;
        }
    }
}


void VoxelGrid::computeOrient3D()
{
    // from m_grid
    m_ori.clear();
    m_ori.resize(m_grid_res[0] * m_grid_res[1] * m_grid_res[2]);
    for (int i = 0; i < m_ori.size(); i++)
    {
        const auto& idxs = m_grid[i];
        if (!idxs.empty())
        {
            Eigen::Vector3f otmp(0.f, 0.f, 0.f);
            Eigen::Vector3f oref = (*m_pcl)[idxs[0]].getNormalVector3fMap().normalized();
            for (int idx : idxs)
            {
                Eigen::Vector3f o = (*m_pcl)[idx].getNormalVector3fMap().normalized();
                if (oref.dot(o) < 0)
                    o = -o;
                otmp += o;
            }
            otmp.normalize();
            if (otmp[2] < 0.f)
                otmp *= -1.f;
            m_ori[i] = otmp;
        }
    }
}

void VoxelGrid::computeDirection3D()
{
    // from m_grid
    m_ori.clear();
    m_ori.resize(m_grid_res[0] * m_grid_res[1] * m_grid_res[2]);
    for (int i = 0; i < m_ori.size(); i++)
    {
        const auto& idxs = m_grid[i];
        Eigen::Vector3f otmp(0.f, 0.f, 0.f);
        if (!idxs.empty())
        {
            for (int idx : idxs)
            {
                Eigen::Vector3f o = (*m_pcl)[idx].getNormalVector3fMap();
                otmp += o;
            }
            otmp /= idxs.size();
        }
        m_ori[i] = otmp;
    }
}

int VoxelGrid::getVoxelGridIndex(const PointT& pt)
{
    int idxx = static_cast<int>(std::floor((pt.x - m_origin[0]) / m_voxel_length));
    int idxy = static_cast<int>(std::floor((pt.y - m_origin[1]) / m_voxel_length));
    int idxz = static_cast<int>(std::floor((pt.z - m_origin[2]) / m_voxel_length));

    int idx = -1;
    if (idxx >= 0 && idxx < m_grid_res[0]
        && idxy >= 0 && idxy < m_grid_res[1]
        && idxz >= 0 && idxz < m_grid_res[2])
        idx = getIndex(idxx, idxy, idxz);
    return idx;
}

int VoxelGrid::getIndex(int idxx, int idxy, int idxz)
{
    int idx = idxz * m_grid_res[0] * m_grid_res[1] + idxy * m_grid_res[0] + idxx;
    return idx;
}

std::vector<int> VoxelGrid::getNeighborVoxels(int grid_idx)
{
    int idxx = grid_idx % m_grid_res[0];
    int idxy = (grid_idx % (m_grid_res[0] * m_grid_res[1])) / m_grid_res[0];
    int idxz = grid_idx / (m_grid_res[0] * m_grid_res[1]);

    int idx = getIndex(idxx, idxy, idxz);
    assert(idx == grid_idx);

    std::vector<int> vec_grid_idx;

    if (idxy + 1 < m_grid_res[1])
    {
        vec_grid_idx.push_back(getIndex(idxx, idxy + 1, idxz));
        if (idxz + 1 < m_grid_res[2])
            vec_grid_idx.push_back(getIndex(idxx, idxy + 1, idxz + 1));
        if (idxz - 1 >= 0)
            vec_grid_idx.push_back(getIndex(idxx, idxy + 1, idxz - 1));
    }

    if (idxy - 1 >= 0)
    {
        vec_grid_idx.push_back(getIndex(idxx, idxy - 1, idxz));
        if (idxz + 1 < m_grid_res[2])
            vec_grid_idx.push_back(getIndex(idxx, idxy - 1, idxz + 1));
        if (idxz - 1 >= 0)
            vec_grid_idx.push_back(getIndex(idxx, idxy - 1, idxz - 1));
    }

    vec_grid_idx.push_back(getIndex(idxx, idxy, idxz));
    if (idxz + 1 < m_grid_res[2])
        vec_grid_idx.push_back(getIndex(idxx, idxy, idxz + 1));
    if (idxz - 1 >= 0)
        vec_grid_idx.push_back(getIndex(idxx, idxy, idxz - 1));

    // idxx + 1
    if (idxx + 1 < m_grid_res[0])
    {
        if (idxy + 1 < m_grid_res[1])
        {
            vec_grid_idx.push_back(getIndex(idxx + 1, idxy + 1, idxz));
            if (idxz + 1 < m_grid_res[2])
                vec_grid_idx.push_back(getIndex(idxx + 1, idxy + 1, idxz + 1));
            if (idxz - 1 >= 0)
                vec_grid_idx.push_back(getIndex(idxx + 1, idxy + 1, idxz - 1));
        }

        if (idxy - 1 >= 0)
        {
            vec_grid_idx.push_back(getIndex(idxx + 1, idxy - 1, idxz));
            if (idxz + 1 < m_grid_res[2])
                vec_grid_idx.push_back(getIndex(idxx + 1, idxy - 1, idxz + 1));
            if (idxz - 1 >= 0)
                vec_grid_idx.push_back(getIndex(idxx + 1, idxy - 1, idxz - 1));
        }

        vec_grid_idx.push_back(getIndex(idxx + 1, idxy, idxz));
        if (idxz + 1 < m_grid_res[2])
            vec_grid_idx.push_back(getIndex(idxx + 1, idxy, idxz + 1));
        if (idxz - 1 >= 0)
            vec_grid_idx.push_back(getIndex(idxx + 1, idxy, idxz - 1));
    }

    // idxx - 1
    if (idxx - 1 >= 0)
    {
        if (idxy + 1 < m_grid_res[1])
        {
            vec_grid_idx.push_back(getIndex(idxx - 1, idxy + 1, idxz));
            if (idxz + 1 < m_grid_res[2])
                vec_grid_idx.push_back(getIndex(idxx - 1, idxy + 1, idxz + 1));
            if (idxz - 1 >= 0)
                vec_grid_idx.push_back(getIndex(idxx - 1, idxy + 1, idxz - 1));
        }

        if (idxy - 1 >= 0)
        {
            vec_grid_idx.push_back(getIndex(idxx - 1, idxy - 1, idxz));
            if (idxz + 1 < m_grid_res[2])
                vec_grid_idx.push_back(getIndex(idxx - 1, idxy - 1, idxz + 1));
            if (idxz - 1 >= 0)
                vec_grid_idx.push_back(getIndex(idxx - 1, idxy - 1, idxz - 1));
        }

        vec_grid_idx.push_back(getIndex(idxx - 1, idxy, idxz));
        if (idxz + 1 < m_grid_res[2])
            vec_grid_idx.push_back(getIndex(idxx - 1, idxy, idxz + 1));
        if (idxz - 1 >= 0)
            vec_grid_idx.push_back(getIndex(idxx - 1, idxy, idxz - 1));
    }

    return vec_grid_idx;
}