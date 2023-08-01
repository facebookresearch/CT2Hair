// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// standard
#include <ctime>
#include <math.h>
#include <algorithm>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <set>

// pcl
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
#include <pcl/common/impl/io.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/pfh.h>

// boost
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// cuda
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

// custom
#include "voxel_grid.h"
#include "clock.h"
#include "cuda_types.h"

#define HAIR_NUM_THREADS 8
#define HAIR_OMP_CHECK_PROGRESS true

class Hair
{
public:
    typedef pcl::PointXYZ PointXYZT;
    typedef pcl::PointXYZLNormal PointT;
    typedef pcl::PointXYZRGBNormal PointColorT;
    typedef std::deque<PointT> Strand;
    typedef std::deque<PointColorT> StrandColor;

public:
    Hair();
    Hair(std::string fn, bool is_strands=false, bool is_color=false);
    ~Hair(){}

    void out2inCloud(bool is_strands=false);
    void meanShift(float nei_radius, int nei_thres, float sigma_e, float sigma_o, float thres_shift, float max_num_shift);
    void meanShiftCUDA(float nei_radius, int nei_thres, float sigma_e, float sigma_o, float thres_shift, float max_num_shift, int gpu_id);
    void genSegmentsFromPointCloud(float nei_radius, float step_size, float thres_orient, float thres_length, float thres_thick);

    void loadRoots(std::string fn);
    void getFinePoorStrands(int k_nearest, float thres_nn_roots_dis, float thres_length);
    void connectPoorStrands();

    void writeOutcloud(std::string fn, bool binary = true);
    void writeOutcloudColor(std::string fn);
    bool writeUSCData(std::string fn);
    bool writeBin(std::string fn);

private:
    // initialize
    void initClouds();

    // 3D Processing
    float getLength(const Strand& strand);
    float orientDifference(const PointT& p1, const PointT& p2);
    float orientDifferenceImpl(const Eigen::Vector3f& v1, const Eigen::Vector3f& v2);
    int linePlaneIntersect(PointT line, PointT pt, PointT& intersect_pt);
    bool forwardEulerStep(const PointT& pt_now, const Eigen::Vector3f& direction, float nei_radius, float step_size, float thres_length, float thres_orient, const std::vector<bool>& is_removed, PointT& pt_next);
    void removePointsCloseToStrand(const Strand& segment, std::vector<bool>& is_removed, float thres_thick);
    void performMeanShift(const std::vector<int>& k_indices, const PointT& pt, float sigma_e, float sigma_o, PointT& pt_new);
    void mergeSegments(Strand& segment1, Strand& segment2, Strand& segment);
    void removeZeroVertices(pcl::PointCloud<PointT>::Ptr cloud);
    void computeStrandDirections();

    Strand reverseStrand(Strand input_strand);
    int strandNearest(Strand strand, PointT point);
    bool poor2fineGenerationFrontBackward(Strand& poor_strand, Strand fine_strand, int nn_idx);
    void poor2fineGenerationBackForward(Strand& poor_strand, Strand fine_strand, int nn_idx);

    // Misc.
    Eigen::Vector3i getColor(int ind);
    float Gaussian(float x, float sigma);
    float rad2deg(float rad);
    void saveStrands2Outcloud();
    void genStrandsFromPointCloud(std::vector<Strand>& strands, pcl::PointCloud<PointT>::Ptr cloud);

public:
    pcl::PointCloud<PointT>::Ptr m_outcloud;
    pcl::PointCloud<PointT>::Ptr m_fine_cloud;
    pcl::PointCloud<PointT>::Ptr m_poor_cloud;
    pcl::PointCloud<PointT>::Ptr m_conn_cloud;

    std::vector<Strand> m_conn_strands;

    // parameters
    float m_thres_angle_roots = 25;
    float m_thres_angle_nn = 25;

    float m_poor_search_radius = 12.f;
    float m_thres_nn_s_count = 0.2f;
    float m_thres_min_length = 5.f;
    float m_thres_grow2root_dis = 1.0f;

private:
    pcl::PointCloud<PointT>::Ptr m_incloud;
    pcl::PointCloud<PointColorT>::Ptr m_incloud_color;
    pcl::KdTreeFLANN<PointT>::Ptr m_tree;
    std::vector<Strand> m_strands;
    VoxelGrid m_grid;

    pcl::PointCloud<PointT>::Ptr m_roots_pos;
    pcl::KdTreeFLANN<PointT>::Ptr m_roots_tree;

    std::vector<Strand> m_fine_strands;
    std::vector<Strand> m_poor_strands;
    std::vector<Strand> m_poor_conn_strands;
    pcl::KdTreeFLANN<PointT>::Ptr m_fine_strands_tree;
    pcl::KdTreeFLANN<PointT>::Ptr m_poor_strands_tree;
};