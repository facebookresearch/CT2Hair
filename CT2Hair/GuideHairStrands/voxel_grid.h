// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <limits>

#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/common/distances.h>
#include <pcl/common/geometry.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>

class VoxelGrid
{
public:
    typedef pcl::PointXYZLNormal PointT;

public:
    VoxelGrid(){}
    VoxelGrid(pcl::PointCloud<PointT>::Ptr pcl, float voxel_length);
    VoxelGrid(pcl::PointCloud<PointT>::Ptr pcl, int W, int H, int D, float minx, float maxx, float miny, float maxy, float minz, float maxz);
    VoxelGrid(pcl::PointCloud<PointT>::Ptr pcl, float voxel_length, float minx, float maxx, float miny, float maxy, float minz, float maxz);
    ~VoxelGrid(){}

    void radiusSearch(const PointT &pt, float radius, std::vector<int> &k_indices);
    void getGridRes(int* grid_res);
    void getGridOrigin(float* origin);
    float getGridVoxelLength();
    int numNonEmptyVoxels();
    Eigen::Vector3f getOrientAt(int idx);
    Eigen::Vector3f getCenterPositionAt(int idx);
    std::vector<int> getPointIndicesAt(int idx);
    std::vector<int> getPointIndicesAt(int idxx, int idxy, int idxz);
    int getIndex(int idxx, int idxy, int idxz);
    int getVoxelGridIndex(const PointT& pt);
    std::vector<std::vector<int>> getGrid() { return m_grid; };
    std::vector<int> getNeighborVoxels(int grid_idx);
    void computeOrient3D();
    void computeDirection3D();

private:
    void initVoxelGrid();
    void initVoxelGrid(int W, int H, int D, float minx, float maxx, float miny, float maxy, float minz, float maxz);
    void initVoxelGrid(float voxel_length, float minx, float maxx, float miny, float maxy, float minz, float maxz);

private:
    pcl::PointCloud<PointT>::Ptr m_pcl;
    float m_voxel_length;
    std::vector<std::vector<int>> m_grid;
    std::vector<Eigen::Vector3f> m_ori;
    int m_grid_res[3];
    float m_origin[3];
};