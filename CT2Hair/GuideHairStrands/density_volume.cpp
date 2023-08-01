// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "density_volume.h"

#define NUM_THREADS 32

DensityVolume::DensityVolume(std::string fn, float voxel_size)
{
    if (!loadStrands(fn))
        exit(0);
    m_voxel_size = voxel_size;
    m_sample_size = m_voxel_size / 2.f;
    printf("Target voxel size is %f.\n", voxel_size);

    m_density_vol = openvdb::FloatGrid::create();

    m_normal_distribution = std::normal_distribution<float>(0.f, 1.f);
}

DensityVolume::DensityVolume(std::string fn_source, std::string fn_target)
{
    openvdb::initialize();
    
    openvdb::GridBase::Ptr grids_base_source;
    openvdb::io::File file_load_source(fn_source);
    file_load_source.open();
    for (openvdb::io::File::NameIterator name_iter = file_load_source.beginName();
        name_iter != file_load_source.endName(); ++name_iter)
    {
        grids_base_source = file_load_source.readGrid(name_iter.gridName());
        break;
    }
    m_source_density = openvdb::gridPtrCast<openvdb::FloatGrid>(grids_base_source);
    file_load_source.close();

    openvdb::GridBase::Ptr grids_base_target;
    openvdb::io::File file_load_target(fn_target);
    file_load_target.open();
    for (openvdb::io::File::NameIterator name_iter = file_load_target.beginName();
        name_iter != file_load_target.endName(); ++name_iter)
    {
        grids_base_target = file_load_target.readGrid(name_iter.gridName());
        break;
    }
    m_target_density = openvdb::gridPtrCast<openvdb::FloatGrid>(grids_base_target);
    file_load_target.close();

    m_num_src_voxels = m_source_density->activeVoxelCount();
    m_num_tgt_voxels = m_target_density->activeVoxelCount();

    printf("Source volume valid voxels number: %d VS target volume valid voxels number: %d.\n",
           m_num_src_voxels, m_num_tgt_voxels);
}

DensityVolume::~DensityVolume()
{

}

bool DensityVolume::loadBinStrands(std::string fn)
{
#ifdef _WIN32
    FILE* f;
    fopen_s(&f, fn.c_str(), "rb");
#else
    FILE* f = fopen(fn.c_str(), "rb");
#endif
    if (!f)
    {
        fprintf(stderr, "Couldn't open %s\n", fn.c_str());
        return false;
    }

    int num_strands = 0;
    fread(&num_strands, 4, 1, f);
    for (int i_strand = 0; i_strand < num_strands; i_strand++)
    {
        int num_points = 0;
        fread(&num_points, 4, 1, f);
        StrandPoints strand_points(num_points, Point(0, 0, 0));
        float dummy = 0.0f;
        for (int j_point = 0; j_point < num_points; j_point++)
        {
            fread(&strand_points[j_point].x, 4, 1, f);
            fread(&strand_points[j_point].y, 4, 1, f);
            fread(&strand_points[j_point].z, 4, 1, f);
            fread(&dummy, 4, 1, f); // nx unused
            fread(&dummy, 4, 1, f); // ny unused
            fread(&dummy, 4, 1, f); // nz unused
            fread(&dummy, 4, 1, f); // label unused
        }
        bool valid_strand = true;
        float strand_length = 0.;
        for (int j_point = 0; j_point < num_points - 1; j_point++)
        {
            float seg_length = pcl::geometry::distance(strand_points[j_point], strand_points[j_point + (size_t)1]);
            if (seg_length < 1.e-4)
            {
                valid_strand = false;
                break;
            }
            strand_length += seg_length;
        }
        if (valid_strand)
        {
            m_input_strds.push_back(strand_points);
            m_input_strds_lens.push_back(strand_length);
            m_num_input_strds++;
            m_num_input_pts += num_points;
        }
    }
    fclose(f);

    printf("Loaded number of strands %d, number of points: %d.\n", m_num_input_strds, m_num_input_pts);
    return true;
}

bool DensityVolume::loadUSCStrands(std::string fn)
{
#ifdef _WIN32
    FILE* f;
    fopen_s(&f, fn.c_str(), "rb");
#else
    FILE* f = fopen(fn.c_str(), "rb");
#endif
    if (!f)
    {
        fprintf(stderr, "Couldn't open %s\n", fn.c_str());
        return false;
    }

    int num_strands = 0;
    fread(&num_strands, 4, 1, f);
    for (int i_strand = 0; i_strand < num_strands; i_strand++)
    {
        int num_points;
        fread(&num_points, 4, 1, f);
        if (num_points == 1)
        {
            Point dummy(0, 0, 0);
            fread(&dummy.x, 4, 1, f);
            fread(&dummy.y, 4, 1, f);
            fread(&dummy.z, 4, 1, f);
        }
        else
        {
            StrandPoints strand_points(num_points, Point(0, 0, 0));
            for (int j_point = 0; j_point < num_points; j_point++)
            {
                fread(&strand_points[j_point].x, 4, 1, f);
                fread(&strand_points[j_point].y, 4, 1, f);
                fread(&strand_points[j_point].z, 4, 1, f);

                strand_points[j_point].getArray3fMap() += Point(0.12f, -1.6f, 0.12f).getArray3fMap();
                strand_points[j_point].getArray3fMap() *= 1000; // m -> mm
            }
            m_input_strds.push_back(strand_points);
            m_num_input_strds++;
            m_num_input_pts += num_points;
        }
    }
    fclose(f);

    printf("Loaded number of strands %d, number of points: %d.\n", m_num_input_strds, m_num_input_pts);
    return true;
}

bool DensityVolume::loadStrands(std::string fn)
{
    if (fn[fn.length() - 3] == 'b') // .bin
        return(loadBinStrands(fn));
    else if (fn[fn.length() - 3] == 'a') // .data
        return(loadUSCStrands(fn));
    
    return false;
}

void DensityVolume::insertVoxels(openvdb::FloatGrid::Accessor vol_accessor, Point center_pos, int morph, float noise_level)
{
    float noise = m_normal_distribution(m_random_generator) * noise_level;

    openvdb::Coord c_xyz(std::roundf(center_pos.x / m_voxel_size),
                         std::roundf(center_pos.y / m_voxel_size),
                         std::roundf(center_pos.z / m_voxel_size));

    int start_idx = -1 * morph;
    int end_idx = morph;
    for (int x = start_idx; x <= end_idx; x++)
    {
        for (int y = start_idx; y <= end_idx; y++)
        {
            for (int z = start_idx; z <= end_idx; z++)
            {
                openvdb::Coord xyz = c_xyz.offsetBy(x, y, z);
                float density = vol_accessor.getValue(xyz);
                if (density == 0.f)
                    density += noise;
                density += 1.f;
                vol_accessor.setValue(xyz, density);

                if (m_max_density <= density)
                    m_max_density = density;
            }
        }
    }

    for (int i = 0; i < 3; i++)
    {
        m_min_xyz[i] = std::min(m_min_xyz[i], c_xyz[i] + start_idx);
        m_max_xyz[i] = std::max(m_max_xyz[i], c_xyz[i] + end_idx);
    }
}

DensityVolume::StrandPoints DensityVolume::densifyStrand(StrandPoints strand, int morph, float gaussian_noise_level)
{
    openvdb::FloatGrid::Accessor vol_accessor = m_density_vol->getAccessor();

    StrandPoints densified_strd;
    int num_strd_pts = strand.size();
    for (int i_p = 0; i_p < num_strd_pts - 1; i_p++)
    {
        float seg_len = pcl::geometry::distance(strand[i_p], strand[i_p + (size_t)1]);
        int num_interp_pts = std::ceil(seg_len / m_sample_size);
        if (num_interp_pts <= 1)
            continue;
        
        Point position_a, position_b, tangent_a, tangent_b, normal;
        position_a = strand[i_p];
        position_b = strand[i_p + (size_t)1];

        if (i_p == 0)
            tangent_a.getArray3fMap() = strand[i_p + (size_t)1].getArray3fMap() - strand[i_p].getArray3fMap();
        else
            tangent_a.getArray3fMap() = strand[i_p].getArray3fMap() - strand[i_p - (size_t)1].getArray3fMap();
        if (i_p == num_strd_pts - 2)
            tangent_b = tangent_a;
        else
            tangent_b.getArray3fMap() = strand[i_p + (size_t)2].getArray3fMap() - strand[i_p + (size_t)1].getArray3fMap();
    
        normal.getArray3fMap() = strand[i_p + (size_t)1].getArray3fMap() - strand[i_p].getArray3fMap();

        tangent_a.getVector3fMap().normalize();
        tangent_b.getVector3fMap().normalize();
        normal.getVector3fMap().normalize();

        float interp_scale = 1.0f / num_interp_pts;
        
        for (int j_p = 0; j_p < num_interp_pts; j_p++)
        {   
            // interp a point
            float t = interp_scale * j_p;
            Point interp_point;
            interp_point.getArray3fMap() = ((powf(t, 3.0f) - 2.0f * powf(t, 2.0f) + t) * tangent_a.getArray3fMap()
                                         + (-2.0f * powf(t, 3.0f) + 3.0f * powf(t, 2.0f)) * normal.getArray3fMap()
                                         + (powf(t, 3.0f) - powf(t, 2.0f)) * tangent_b.getArray3fMap()) * seg_len + position_a.getArray3fMap();

            insertVoxels(vol_accessor, interp_point, morph, gaussian_noise_level);
        }
    }
    // for the last point
    Point last_point = strand[num_strd_pts - (size_t)1];
    insertVoxels(vol_accessor, last_point, morph, gaussian_noise_level);

    return densified_strd;
}

void DensityVolume::calculateDensityVolume(int morph_width, int gaussian_width, float gaussian_noise_level)
{
    for (int i_s = 0; i_s < m_num_input_strds; i_s++)
    {
        StrandPoints strand = m_input_strds[i_s];
        StrandPoints densified_strd = densifyStrand(strand, morph_width, gaussian_noise_level);
        printf("\rStrand: %d/%d", i_s, m_num_input_strds);
    }
    printf(" finished!, max density value is: %f\n", m_max_density);
    // std::cout << "            volume bound: " << m_min_xyz << ", " << m_max_xyz << std::endl;
    
    size_t distrib_size = (size_t)m_max_density + (size_t)1;
    m_density_distrib = new int[distrib_size];
    // printVolDistrib(distrib_size);
    
    if (gaussian_width > 0)
    {
        openvdb::tools::Filter<openvdb::FloatGrid> grid_filter(*m_density_vol);
        grid_filter.gaussian(gaussian_width);
        // printVolDistrib(distrib_size);
    }
}

void DensityVolume::printVolDistrib(size_t distrib_size)
{
    for (int i = 0; i < distrib_size; i++)
        m_density_distrib[i] = 0;
    for(openvdb::FloatGrid::ValueOnIter iter = m_density_vol->beginValueOn(); iter; ++iter)
    {
        float value = iter.getValue();
        m_density_distrib[(size_t)value] += 1;
    }
    for (int i = 0; i < distrib_size; i++)
        printf("%d\n", m_density_distrib[i]);  
    printf("\n\n\n");
}

void DensityVolume::saveDenseVoxels(std::string fn)
{
    typedef unsigned short ushort;
    typedef unsigned long long int ulong;
    openvdb::Coord voxels_size = m_max_xyz - m_min_xyz;
    std::cout << "Dense voxels size is: " << voxels_size;
    ulong voxels_count = (ulong)voxels_size[0] * (ulong)voxels_size[1] * (ulong)voxels_size[2];
    ushort* dense_voxels = new ushort[voxels_count];

    openvdb::FloatGrid::Accessor vdb_acc = m_density_vol->getAccessor();

#pragma omp parallel for num_threads(NUM_THREADS)
    for (int x = 0; x < voxels_size[0] - 20; x++)
    {
        for (int y = 0; y < voxels_size[1]; y++)
        {
            for (int z = 0; z < voxels_size[2]; z++)
            {
                ulong vox_ind = (ulong)x * (ulong)(voxels_size[1] * voxels_size[2])
                              + (ulong)y * (ulong)voxels_size[2]
                              + (ulong)z;
                openvdb::Coord vdb_ind(x + m_min_xyz[0], y + m_min_xyz[1], z + m_min_xyz[2]);
                // printf("\rVoxels: %llu/%llu", vox_ind, voxels_count);
                dense_voxels[vox_ind] = (ushort)(vdb_acc.getValue(vdb_ind));
            }
        }
    }

    FILE* f = fopen(fn.c_str(), "wb");

    if (!f)
    {
        fprintf(stderr, "Couldn't open %s\n", fn.c_str());
        return;
    }

    fwrite(dense_voxels, sizeof(ushort), voxels_count, f);

    fclose(f);

    printf(", dense voxels generation finished!\n");
}

void DensityVolume::saveDensityVolume(std::string fn)
{
    m_density_vol->setGridClass(openvdb::GRID_LEVEL_SET);
    m_density_vol->setName("density");
    openvdb::io::File file(fn);

    openvdb::GridPtrVec grids;
    grids.push_back(m_density_vol);

    file.write(grids);
    file.close();
}