// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "hair_vdb.h"
#include "cuda_types.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <vector_types.h>   // from cuda library include
#include <openvdb/tools/Clip.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/math/BBox.h>
#include <Eigen/Core>
#include <pcl/filters/extract_indices.h>

#define NUM_THREADS 32
#define MAX_NUM_OUT_PTS 89478486

HairVDB::HairVDB(std::string vdb_name, float voxel_size, std::string roots_name)
{
    printf("Loading vdb and calculating gradients...\n");

    if (vdb_name != "")
    {
        openvdb::initialize();
        openvdb::io::File file_load(vdb_name);
        file_load.open();

        for (openvdb::io::File::NameIterator name_iter = file_load.beginName();
            name_iter != file_load.endName(); ++name_iter)
        {
            m_density_base = file_load.readGrid(name_iter.gridName());
            break;
        }

        m_density = openvdb::gridPtrCast<openvdb::FloatGrid>(m_density_base);
        file_load.close();
        
        m_gradient = openvdb::tools::gradient(*m_density);
        m_num_active_voxels = m_density->activeVoxelCount();
    }

    printf("\tnumber of valid voxels: %d; ", m_num_active_voxels);

    m_grads = pcl::PointCloud<Point>::Ptr(new pcl::PointCloud<Point>);
    m_pts = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    m_roots = pcl::PointCloud<Point>::Ptr(new pcl::PointCloud<Point>);
    m_coords_tree = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    m_roots_tree = pcl::KdTreeFLANN<Point>::Ptr(new pcl::KdTreeFLANN<Point>);

    (*m_grads).reserve(m_num_active_voxels);
    (*m_pts).reserve(m_num_active_voxels);

    m_voxel_size = voxel_size;

    if(roots_name != "")
        loadHairRoots(roots_name);
}

HairVDB::~HairVDB()
{

}

void HairVDB::getPoints()
{
    printf("Getting points initialized...\n");
    int points_count = 0;
    for (openvdb::Vec3SGrid::ValueOnCIter iter = m_gradient->cbeginValueOn(); iter.test(); ++iter)
    {
        points_count++;
        openvdb::Vec3f gradient = *iter;
        openvdb::math::Coord coord = iter.getCoord();
        
        if (m_num_roots_pts > 0)
        {
            Point p(coord.x() * m_voxel_size, coord.y() * m_voxel_size, coord.z() * m_voxel_size);
            pcl::Indices nn_idx;
            std::vector<float> nn_dis;
            m_roots_tree->radiusSearch(p, m_wig_dis_thres, nn_idx, nn_dis);

            if(nn_dis.size() > 0)
                continue;
        }

        Point grad(gradient.x(), gradient.y(), gradient.z());
        PointT pt(coord.x(), coord.y(), coord.z(), 0, 0.f, 0.f, 0.f);
        (*m_grads).push_back(grad);
        (*m_pts).push_back(pt);
        if (points_count % 1000 == 0)
            printf("\r\tprogress: %2.2f%%", ((float)points_count / (float)m_num_active_voxels * 100.f));
    }
    m_num_hair_pts = (*m_grads).size();
    
    // Kd tree build and search also slow (more than 200s on desktop, 60s on server)
    m_coords_tree->setInputCloud(m_pts);

    std::cout << "\r\tnumber of hair points: " << m_num_hair_pts << "; ";
}

Eigen::Vector3d get_eigen_vector_with_smallest_eigen_value(Eigen::Matrix3d A)
{
    const double eps = 1e-20;
    double eigen_value_prev = 0.;
    double eigen_value_new = 0.;

    int iter = 0;
    const int iter_max = 100;

    if (A.determinant() == 0.)
        return Eigen::Vector3d(0., 0., 0.);

    Eigen::Vector3d x(1., 1., 1.);
    float diff = 1.f;
    while ((iter < iter_max) && (diff > eps))
    {
        x = A.inverse() * x;
        x.normalize();
        eigen_value_new = x.dot(A * x);
        diff = abs(eigen_value_new - eigen_value_prev);
        eigen_value_prev = eigen_value_new;
        iter++;
    }
    return x;
}

void HairVDB::calculateOriens()
{
    printf("Calculating orientations...\n");
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i_pt = 0; i_pt < m_num_hair_pts; i_pt++)
    {
        pcl::Indices nn_idx;
        std::vector<float> nn_dis;
        m_coords_tree->radiusSearch((*m_pts)[i_pt], m_bbox_radius, nn_idx, nn_dis);

        int num_nei = nn_dis.size();
        int num_valid_nei = 0;
        Eigen::MatrixX3d A_all(num_nei, 3);
        Point zero_point(0.f, 0.f, 0.f);
        for (int j_nei = 0; j_nei < num_nei; j_nei++)
        {
            Point grad = (*m_grads)[nn_idx[j_nei]];
            float dis = pcl::euclideanDistance<Point, Point>(grad, zero_point);
            if (dis <= 0)
                continue;

            A_all(num_valid_nei, 0) = grad.x;
            A_all(num_valid_nei, 1) = grad.y;
            A_all(num_valid_nei, 2) = grad.z;
            num_valid_nei++;
        }

        Eigen::MatrixX3d A(num_valid_nei, 3);
        A = A_all(Eigen::seqN(0, num_valid_nei), Eigen::all);

        if (num_valid_nei < 6)
            continue;
        
        Eigen::Matrix3d AtA = A.transpose() * A;    // AtA: 3 x 3 matrix
        Eigen::Vector3d x = get_eigen_vector_with_smallest_eigen_value(AtA);

        if (x == Eigen::Vector3d(0., 0., 0.))
            continue;

        Eigen::VectorXd Ax(num_valid_nei);
        Ax = A * x;
        Eigen::VectorXd w(num_valid_nei);
        w = Ax.cwiseInverse();
        A = A.array().colwise() * (w.array() * w.array());
        
        double A_max = -DBL_MAX;
        for (int j_col = 0; j_col < 3; j_col++)
            if (A_max < A.col(j_col).maxCoeff())
                A_max = A.col(j_col).maxCoeff();
        A = A / A_max;

        AtA = A.transpose() * A;
        x = get_eigen_vector_with_smallest_eigen_value(AtA);
        (*m_pts)[i_pt].normal_x = x(0);
        (*m_pts)[i_pt].normal_y = x(1);
        (*m_pts)[i_pt].normal_z = x(2);
    }
    printf("\t");
}

void HairVDB::exportPts(std::string file_name, bool exp_bin)
{
    // filtering out zero orientations
    pcl::PointCloud<PointT>::Ptr valid_pts;
    valid_pts = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    (*valid_pts).reserve(m_num_hair_pts);

    for (int i_pt = 0; i_pt < m_num_hair_pts; i_pt++)
    {
        PointT pt = (*m_pts)[i_pt];
        if (pt.normal_x == 0 && pt.normal_y == 0 && pt.normal_z == 0)
            continue;
        
        pt.x = pt.x * m_voxel_size;
        pt.y = pt.y * m_voxel_size;
        pt.z = pt.z * m_voxel_size;
        (*valid_pts).push_back(pt);
    }
    
    auto gen = std::mt19937{ std::random_device{}() };
    pcl::PointCloud<PointT>::Ptr sampled_pts;
    sampled_pts = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    (*sampled_pts).reserve(m_num_hair_pts);
    std::sample((*valid_pts).begin(), (*valid_pts).end(), std::back_inserter(*sampled_pts), MAX_NUM_OUT_PTS, gen);

    printf("Exporting points...\n");
    pcl::PLYWriter w;
    w.write<PointT>(file_name, *sampled_pts, true, false);

    // save original numbers of points
    if (exp_bin)
        writeBinPts(file_name.replace(file_name.end() - 4, file_name.end(), ".bin"), valid_pts);
    printf("\t");
}

void HairVDB::exportPtsWithValue(std::string file_name, float voxel_value_max, bool exp_bin)
{
    // filtering out zero orientations
    pcl::PointCloud<PointV>::Ptr valid_pts;
    valid_pts = pcl::PointCloud<PointV>::Ptr(new pcl::PointCloud<PointV>);
    (*valid_pts).reserve(m_num_hair_pts);

    openvdb::FloatGrid::Accessor density_accessor = m_density->getAccessor();
    for (int i_pt = 0; i_pt < m_num_hair_pts; i_pt++)
    {
        PointT pt = (*m_pts)[i_pt];
        if (pt.normal_x == 0 && pt.normal_y == 0 && pt.normal_z == 0)
            continue;

        openvdb::Coord xyz_idx(std::round(pt.x), std::round(pt.y), std::round(pt.z));
        float density_value = density_accessor.getValue(xyz_idx);       
        if (density_value == 0)
            continue;
        
        pt.x = pt.x * m_voxel_size;
        pt.y = pt.y * m_voxel_size;
        pt.z = pt.z * m_voxel_size;

        unsigned int v_r = std::min(density_value, voxel_value_max) / voxel_value_max * 255;
        PointV pt_v(pt.x, pt.y, pt.z, v_r, 0, 0, pt.normal_x, pt.normal_y, pt.normal_z);
        (*valid_pts).push_back(pt_v);
    }
    
    auto gen = std::mt19937{ std::random_device{}() };
    pcl::PointCloud<PointV>::Ptr sampled_pts;
    sampled_pts = pcl::PointCloud<PointV>::Ptr(new pcl::PointCloud<PointV>);
    (*sampled_pts).reserve(m_num_hair_pts);
    std::sample((*valid_pts).begin(), (*valid_pts).end(), std::back_inserter(*sampled_pts), MAX_NUM_OUT_PTS, gen);

    printf("Exporting points...\n");
    pcl::PLYWriter w;
    w.write<PointV>(file_name, *sampled_pts, true, false);

    // save original numbers of points
    if (exp_bin)
        writeBinPtsWithValue(file_name.replace(file_name.end() - 4, file_name.end(), ".bin"), valid_pts);
    printf("\t");
}

void HairVDB::loadOriens(std::string file_name)
{
    if (file_name[file_name.length() - 1] == 'y') {
        pcl::io::loadPLYFile(file_name.c_str(), *m_pts);
        std::cout << "Reading " << (*m_pts).size() << " points," << std::endl;
    }
    else
        loadBinOriens(file_name);
}

void HairVDB::loadBinOriens(std::string file_name)
{
#ifdef _WIN32
    FILE* f;
    fopen_s(&f, file_name.c_str(), "rb");
#else
    FILE* f = fopen(file_name.c_str(), "rb");
#endif
    if (!f)
        fprintf(stderr, "Couldn't open %s\n", file_name.c_str());

    int num_pts;

    fread(&num_pts, 4, 1, f);
    (*m_pts).reserve(num_pts);
    std::cout << "Reading " << num_pts << " points." << std::endl;

    for (unsigned int i = 0; i < num_pts; i++) {
        float x, y, z, nx, ny, nz;
        uint32_t label = 0;
        fread(&x, 4, 1, f);
        fread(&y, 4, 1, f);
        fread(&z, 4, 1, f);
        fread(&nx, 4, 1, f);
        fread(&ny, 4, 1, f);
        fread(&nz, 4, 1, f);

        PointT pt(x, y, z, label, nx, ny, nz);
        (*m_pts).push_back(pt);
    }
    fclose(f);
}

void HairVDB::addValueToOriens(std::string file_name, float voxel_value_max)
{
    pcl::PointCloud<PointV>::Ptr pts_v;
    pts_v = pcl::PointCloud<PointV>::Ptr(new pcl::PointCloud<PointV>);
    int num_oriens = (*m_pts).size();
    (*pts_v).reserve(num_oriens);

    openvdb::FloatGrid::Accessor density_accessor = m_density->getAccessor();

    std::cout << "Add values," << std::endl;
    for (int i_pt = 0; i_pt < num_oriens; i_pt++)
    {
        PointT pt = (*m_pts)[i_pt];
        float x_idx = pt.x / m_voxel_size;
        float y_idx = pt.y / m_voxel_size;
        float z_idx = pt.z / m_voxel_size;

        openvdb::Coord xyz_idx(std::round(x_idx), std::round(y_idx), std::round(z_idx));
        float density_value = density_accessor.getValue(xyz_idx);
        
        if (density_value == 0)
            continue;
        
        unsigned int v_r = std::min(density_value, voxel_value_max) / voxel_value_max * 255;
        PointV pt_v(pt.x, pt.y, pt.z, v_r, 0, 0, pt.normal_x, pt.normal_y, pt.normal_z);
        (*pts_v).push_back(pt_v);
    }

    std::cout << "\tnumber of exported points: " << (*pts_v).size() << std::endl;
    pcl::PLYWriter w;
    w.write<PointV>(file_name, *pts_v, true, false);
}

void HairVDB::loadHairRoots(std::string file_name)
{
    pcl::io::loadPLYFile(file_name.c_str(), *m_roots);
    m_num_roots_pts = (*m_roots).size();
    m_roots_tree->setInputCloud(m_roots);
}

void HairVDB::removeWignetInOriens(std::string file_name, bool exp_bin)
{
    if (m_num_roots_pts <= 0)
    {
        std::cout << "Load roots firstly!" << std::endl;
        exit(0);
    }

    pcl::PointCloud<PointT>::Ptr pts_no_wig;
    pts_no_wig = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    int num_oriens = (*m_pts).size();
    (*pts_no_wig).reserve(num_oriens);

    for (int i_pt = 0; i_pt < num_oriens; i_pt++)
    {
        PointT pt = (*m_pts)[i_pt];

        Point p(pt.x, pt.y, pt.z);
        pcl::Indices nn_idx;
        std::vector<float> nn_dis;
        m_roots_tree->radiusSearch(p, m_wig_dis_thres, nn_idx, nn_dis);
        
        if(nn_dis.size() == 0)
            (*pts_no_wig).push_back(pt);
    }
    

    std::cout << "Removing done and saving points" << std::endl;
    pcl::PLYWriter w;
    if ((*pts_no_wig).size() > MAX_NUM_OUT_PTS)
    {
        auto gen = std::mt19937{ std::random_device{}() };
        pcl::PointCloud<PointT>::Ptr sampled_pts;
        sampled_pts = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
        (*sampled_pts).reserve(m_num_hair_pts);
        std::sample((*pts_no_wig).begin(), (*pts_no_wig).end(), std::back_inserter(*sampled_pts), MAX_NUM_OUT_PTS, gen);
        // sampling points to the max number
        w.write<PointT>(file_name, *sampled_pts, true, false);

        if (exp_bin)
            writeBinPts(file_name.replace(file_name.end() - 4, file_name.end(), ".bin"), pts_no_wig);
    }
    else
        w.write<PointT>(file_name, *pts_no_wig, true, false);
}

bool HairVDB::writeBinPts(std::string file_name, pcl::PointCloud<PointT>::Ptr pcl)
{
    FILE* f = fopen(file_name.c_str(), "wb");
    if (!f)
    {
        fprintf(stderr, "Couldn't open %s\n", file_name.c_str());
        return false;
    }
    
    int num_pts = (*pcl).size();
    std::cout << "\twrting points to a binary file,\n\tnumber of points: " << num_pts << std::endl;
    fwrite(&num_pts, 4, 1, f);
    for (int i = 0; i < num_pts; i++)
    {
        fwrite(&(*pcl)[i].x, 4, 1, f);
        fwrite(&(*pcl)[i].y, 4, 1, f);
        fwrite(&(*pcl)[i].z, 4, 1, f);
        fwrite(&(*pcl)[i].normal_x, 4, 1, f);
        fwrite(&(*pcl)[i].normal_y, 4, 1, f);
        fwrite(&(*pcl)[i].normal_z, 4, 1, f);
    }

    fclose(f);
    return true;
}

bool HairVDB::writeBinPtsWithValue(std::string file_name, pcl::PointCloud<PointV>::Ptr pcl)
{
    FILE* f = fopen(file_name.c_str(), "wb");
    if (!f)
    {
        fprintf(stderr, "Couldn't open %s\n", file_name.c_str());
        return false;
    }
    
    int num_pts = (*pcl).size();
    std::cout << "\twrting points to a binary file,\n\tnumber of points: " << num_pts << std::endl;
    fwrite(&num_pts, 4, 1, f);
    for (int i = 0; i < num_pts; i++)
    {
        fwrite(&(*pcl)[i].x, 4, 1, f);
        fwrite(&(*pcl)[i].y, 4, 1, f);
        fwrite(&(*pcl)[i].z, 4, 1, f);
        float density = (float)(*pcl)[i].r / 255.;
        fwrite(&density, 4, 1, f);
        fwrite(&(*pcl)[i].normal_x, 4, 1, f);
        fwrite(&(*pcl)[i].normal_y, 4, 1, f);
        fwrite(&(*pcl)[i].normal_z, 4, 1, f);
    }

    fclose(f);
    return true;
}

void HairVDB::releaseMem()
{
    // vdbs
    m_gradient.reset();

    // pcl
    m_grads.reset();
    m_roots.reset();
    m_roots_tree.reset();
    m_coords_tree.reset();
}