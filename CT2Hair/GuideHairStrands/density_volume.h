// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>
#include <limits.h>
#include <random>

#include <openvdb/openvdb.h>
#include <openvdb/tools/Filter.h>

#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/geometry.h>

class DensityVolume
{
public:
    typedef pcl::PointXYZ Point;
    typedef pcl::PointXYZLNormal PointT;
    typedef std::vector<std::vector<Point>> StrandsPoints;
    typedef std::vector<Point> StrandPoints;

    DensityVolume(std::string fn, float voxel_size);
    DensityVolume(std::string fn_source, std::string fn_target);
    ~DensityVolume();

    void calculateDensityVolume(int morph_width, int gaussian_width, float gaussian_noise_level);
    void saveDensityVolume(std::string fn);
    void saveDenseVoxels(std::string fn);

private:
    bool loadBinStrands(std::string fn);
    bool loadUSCStrands(std::string fn);
    bool loadStrands(std::string fn);
    void insertVoxels(openvdb::FloatGrid::Accessor accessor, Point center_pos, int morph, float noise_level);
    void printVolDistrib(size_t distrib_size);
    StrandPoints densifyStrand(StrandPoints strand, int morph, float gaussian_noise_level);

private:
    int m_num_input_strds = 0;
    int m_num_input_pts = 0;
    StrandsPoints m_input_strds;
    std::vector<float> m_input_strds_lens;
    
    float m_voxel_size = 0.f;
    float m_sample_size = 0.f;
    float m_max_density = 0.f;
    int* m_density_distrib;

    openvdb::FloatGrid::Ptr m_density_vol;
    openvdb::FloatGrid::Ptr m_source_density;
    openvdb::FloatGrid::Ptr m_target_density;
    int m_num_src_voxels = 0;
    int m_num_tgt_voxels = 0;

    openvdb::Coord m_min_xyz = openvdb::Coord(INT_MAX, INT_MAX, INT_MAX);
    openvdb::Coord m_max_xyz = openvdb::Coord(INT_MIN, INT_MIN, INT_MIN);

    std::default_random_engine m_random_generator;
    std::normal_distribution<float> m_normal_distribution;
};