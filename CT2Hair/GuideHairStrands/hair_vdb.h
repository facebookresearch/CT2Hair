// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <openvdb/openvdb.h>

#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/io/ply_io.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/common/geometry.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/kdtree/kdtree_flann.h>

class HairVDB
{
public:
    typedef pcl::PointXYZ Point;
    typedef pcl::PointXYZLNormal PointT;
    typedef pcl::PointXYZRGBNormal PointV;
    HairVDB(std::string vdb_name, float voxel_size=0.125f, std::string roots_name="");
    ~HairVDB();

    void getPoints();
    void calculateOriens();
    void exportPts(std::string file_name, bool exp_bin=true);
    void exportPtsWithValue(std::string file_name, float voxel_value_max, bool exp_bin=true);

    void loadOriens(std::string file_name);
    void removeWignetInOriens(std::string file_name, bool exp_bin=true);
    void addValueToOriens(std::string file_name, float voxel_value_max=1.0f);

    void releaseMem();

private:
    openvdb::GridBase::Ptr m_density_base;
    openvdb::FloatGrid::Ptr m_density;
    openvdb::Vec3SGrid::Ptr m_gradient;
    pcl::KdTreeFLANN<PointT>::Ptr m_coords_tree;
    pcl::KdTreeFLANN<Point>::Ptr m_roots_tree;
    int m_num_roots_pts = 0;

public:
    int m_bbox_radius = 3;
    float m_voxel_size = 0.125f;
    int m_num_active_voxels = 0;
    int m_num_hair_pts = 0;
    pcl::PointCloud<Point>::Ptr m_grads;
    pcl::PointCloud<PointT>::Ptr m_pts;
    pcl::PointCloud<Point>::Ptr m_roots;

    float m_wig_dis_thres = 2.4f;

private:
    void loadHairRoots(std::string file_name);
    void loadBinOriens(std::string file_name);
    bool writeBinPts(std::string file_name, pcl::PointCloud<PointT>::Ptr pcl);
    bool writeBinPtsWithValue(std::string file_name, pcl::PointCloud<PointV>::Ptr pcl);
};