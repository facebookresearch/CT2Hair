// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "hair.h"

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
    float max_num_shift);

#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

inline void cudaGetLastErrorAndSync( const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : cutilCheckMsg() CUTIL CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }

	err = cudaDeviceSynchronize();
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : cutilCheckMsg cudaDeviceSynchronize error: %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

Hair::Hair()
{
    initClouds();
}

Hair::Hair(std::string fn, bool is_strands, bool is_color)
{
    initClouds();
    if (is_color)
    {
        pcl::io::loadPLYFile(fn.c_str(), *m_incloud_color);
        pcl::copyPointCloud(*m_incloud_color, *m_incloud);
        m_incloud_color.reset();
    }
    else
        pcl::io::loadPLYFile(fn.c_str(), *m_incloud);

    if (is_strands)
    {
        genStrandsFromPointCloud(m_strands, m_incloud);
        computeStrandDirections();
    }

    m_tree->setInputCloud(m_incloud);
}

void Hair::initClouds()
{
    m_incloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    m_incloud_color = pcl::PointCloud<PointColorT>::Ptr(new pcl::PointCloud<PointColorT>);
    m_outcloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    m_tree = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
}

void Hair::out2inCloud(bool is_strands)
{
    m_incloud->clear();
    pcl::copyPointCloud(*m_outcloud, *m_incloud);

    if (is_strands)
    {
        genStrandsFromPointCloud(m_strands, m_incloud);
        computeStrandDirections();
    }

    m_tree.reset();
    m_tree = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    m_tree->setInputCloud(m_incloud);
}

void Hair::meanShift(float nei_radius, int nei_thres, float sigma_e, float sigma_o, float thres_shift, float max_num_shift)
{
    *m_outcloud = *m_incloud;

    int pnumber = (int)m_incloud->size();
    int cntloop = 0;

#pragma omp parallel for num_threads(HAIR_NUM_THREADS) 
    for (int point_id = 0; point_id < pnumber; ++point_id)
    {
#if HAIR_OMP_CHECK_PROGRESS
#pragma omp atomic
        cntloop++;

        if (cntloop % 200000 == 0 || cntloop == pnumber)
        {
#pragma omp critical
            {
                fprintf(stdout, "Hair Filtering:    %d / %d (%.2f%%)\n", cntloop, pnumber, 100.f * cntloop / pnumber);
                fflush(stdout);
            }
        }
#endif

        // current point
        PointT& pt = (*m_incloud)[point_id];

        // Neighbors containers
        std::vector<int> k_indices;
        std::vector<float> k_sqr_distances;
        m_tree->radiusSearch(point_id, nei_radius, k_indices, k_sqr_distances);
        
        // simple noise removal
        if (k_indices.size() < nei_thres)
        {
            (*m_outcloud)[point_id] = PointT();
            continue;
        }

        // mean shift
        PointT pt_old, pt_new;
        float shift = std::numeric_limits<float>::max();
        int count = 0;
        pt_old = pt;
        while (shift > thres_shift && count < max_num_shift)
        {
            performMeanShift(k_indices, pt_old, sigma_e, sigma_o, pt_new);
            shift = pcl::geometry::distance(pt_old, pt_new);
            pt_old = pt_new;
            count++;
        }

        if (count == max_num_shift)
            printf("not converged point: [%f, %f, %f]\n", pt.x, pt.y, pt.z);

        // output
        (*m_outcloud)[point_id] = pt_new;
    }

    removeZeroVertices(m_outcloud);
}

void Hair::meanShiftCUDA(float nei_radius, int nei_thres, float sigma_e, float sigma_o, float thres_shift, float max_num_shift, int gpu_id)
{
    Clock clk;

    int gpu_count = 1;
    cudaGetDeviceCount(&gpu_count);
    if (gpu_id >= 0 && gpu_id < gpu_count)
    {
        cudaSetDevice(gpu_id);
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    }
    else
    {
        printf("invalid GPU ID: %d (# of GPUs: %d)\n", gpu_id, gpu_count);
        return;
    }

    int numpts = (*m_incloud).size();

    // set voxel grid in GPU global memory
    VoxelGrid grid(m_incloud, nei_radius);
    int grid_res_tmp[3];
    grid.getGridRes(&grid_res_tmp[0]);
    int3 grid_res;
    grid_res.x = grid_res_tmp[0];
    grid_res.y = grid_res_tmp[1];
    grid_res.z = grid_res_tmp[2];

    int numvoxels = grid_res.x * grid_res.y * grid_res.z;

    int2* d_grid_idx;
    int* d_grid_data;
    cudaSafeCall(cudaMalloc(&d_grid_idx, numvoxels * sizeof(int2)));
    cudaSafeCall(cudaMalloc(&d_grid_data, numpts * sizeof(int)));

    int2* h_grid_idx = (int2*)malloc(numvoxels * sizeof(int2));
    int* h_grid_data = (int*)malloc(numpts * sizeof(int2));
    int current_idx = 0;
    int current_idx2 = 0;
    for (int i = 0; i < numvoxels; ++i)
    {
        // grid idx
        std::vector<int> pointidx = grid.getPointIndicesAt(i);
        int length = pointidx.size();
        h_grid_idx[i].x = current_idx;
        h_grid_idx[i].y = length;
        current_idx += length;

        // grid data
        for (auto ptidx : pointidx)
            h_grid_data[current_idx2++] = ptidx;
    }
    assert(current_idx == numpts);
    assert(current_idx2 == numpts);

    cudaSafeCall(cudaMemcpy(d_grid_idx, h_grid_idx, numvoxels * sizeof(int2), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_grid_data, h_grid_data, numpts * sizeof(int), cudaMemcpyHostToDevice));

    // set point cloud in GPU global memory
    Point3D* d_incloud, * d_outcloud;
    cudaSafeCall(cudaMalloc(&d_incloud, numpts * sizeof(Point3D)));
    cudaSafeCall(cudaMalloc(&d_outcloud, numpts * sizeof(Point3D)));

    Point3D* h_incloud = (Point3D*)malloc(numpts * sizeof(Point3D));
    Point3D* h_outcloud = (Point3D*)malloc(numpts * sizeof(Point3D));
#pragma omp parallel for num_threads(HAIR_NUM_THREADS)
    for (int i = 0; i < numpts; ++i)
    {
        PointT pt = (*m_incloud)[i];
        Point3D& pt_tmp = h_incloud[i];
        pt_tmp.pos = make_float3(pt.x, pt.y, pt.z);
        pt_tmp.dir = make_float3(pt.normal_x, pt.normal_y, pt.normal_z);
        pt_tmp.grid_idx = pt.label;
    }
    cudaSafeCall(cudaMemcpy(d_incloud, h_incloud, numpts * sizeof(Point3D), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_outcloud, h_incloud, numpts * sizeof(Point3D), cudaMemcpyHostToDevice));

    // launch CUDA mean-shift 
    clk.tick();
    launchMeanShiftCUDA(d_incloud, d_outcloud, d_grid_idx, d_grid_data, grid_res, numvoxels, numpts, nei_radius, nei_thres, sigma_e, sigma_o, thres_shift, max_num_shift);
    cudaGetLastErrorAndSync("launch MeanShiftCUDA error", __FILE__, __LINE__);
    printf("MeanShift with CUDA: %.2f sec.\n", clk.tock());

    // copy data from device to host
    cudaSafeCall(cudaMemcpy(h_outcloud, d_outcloud, numpts * sizeof(Point3D), cudaMemcpyDeviceToHost));
    for (int i = 0; i < numpts; ++i)
    {
        Point3D& pt_tmp = h_outcloud[i];
        PointT pt;
        pt.getVector3fMap() = Eigen::Vector3f(pt_tmp.pos.x, pt_tmp.pos.y, pt_tmp.pos.z);
        pt.getNormalVector3fMap() = Eigen::Vector3f(pt_tmp.dir.x, pt_tmp.dir.y, pt_tmp.dir.z);

        //////////////////////////////////////////////////////////////////////////////
        // This is an ad-hoc solution to handle a bug
        // We want to check if the point is [0,0,0]
        // For some unknown reason (maybe GPU <-> CPU data conversion issue?), 
        // [0,0,0] is [0 0 1.777662e+22]
        //////////////////////////////////////////////////////////////////////////////
        const auto& pttmp = pt.getVector3fMap();
        const auto& dirtmp = pt.getNormalVector3fMap();
        if (pttmp[0] == 0.f || pttmp[1] == 0.f || pttmp[2] == 0.f)
            continue;
        if (dirtmp.norm() < 0.9f || dirtmp.norm() > 1.1f)
            continue;

        (*m_outcloud).push_back(pt);
    }

    removeZeroVertices(m_outcloud);

    free(h_incloud);
    free(h_outcloud);
    free(h_grid_idx);
    free(h_grid_data);
    cudaSafeCall(cudaFree(d_incloud));
    cudaSafeCall(cudaFree(d_outcloud));
    cudaSafeCall(cudaFree(d_grid_idx));
    cudaSafeCall(cudaFree(d_grid_data));
}

void Hair::genSegmentsFromPointCloud(float nei_radius, float step_size, float thres_orient, float thres_length, float thres_thick)
{
    int thres_cutoff = 10000;

    int pnumber = (int)m_incloud->size();
    std::vector<bool> is_removed(pnumber, false);
    uint32_t segment_id = 0;

    m_outcloud->clear();

    for (int point_id = 0; point_id < pnumber; ++point_id)
    {
        if (point_id % 10000 == 0)
            printf("\rProcess... %d / %d (%.2f%%)", point_id, pnumber, 100.f * point_id / pnumber);

        if (is_removed[point_id])
            continue;

        const Eigen::Vector3f& pttmp = (*m_incloud)[point_id].getVector3fMap();
        const Eigen::Vector3f& dirtmp = (*m_incloud)[point_id].getNormalVector3fMap();
        if (pttmp[0] == 0.f || pttmp[1] == 0.f || pttmp[2] == 0.f)
            continue;
        if (dirtmp.norm() < 0.9f || dirtmp.norm() > 1.1f)
            continue;

        PointT pt_now, pt_next;
        Eigen::Vector3f direction;
        Strand segment, segment1, segment2;
        bool found;
        int cnt = 0;

        // one direction
        pt_now = (*m_incloud)[point_id];
        direction = pt_now.getNormalVector3fMap().normalized();
        do
        {
            found = forwardEulerStep(pt_now, direction, nei_radius, step_size, thres_length, thres_orient, is_removed, pt_next);
            if (found)
            {
                segment1.push_back(pt_next);
                direction = pt_next.getNormalVector3fMap();
                pt_now = pt_next;
                cnt++;
            }
        } while (found && cnt < thres_cutoff);
        if (cnt == thres_cutoff)
        {
            segment1.clear();
        }

        // the other direction
        pt_now = (*m_incloud)[point_id];
        direction = -pt_now.getNormalVector3fMap().normalized();
        cnt = 0;
        do
        {
            found = forwardEulerStep(pt_now, direction, nei_radius, step_size, thres_length, thres_orient, is_removed, pt_next);
            if (found)
            {
                segment2.push_back(pt_next);
                direction = pt_next.getNormalVector3fMap();
                pt_now = pt_next;
                cnt++;
            }
        } while (found && cnt < thres_cutoff);
        if (cnt == thres_cutoff)
        {
            segment2.clear();
        }

        // merge
        mergeSegments(segment1, segment2, segment);

        // save points
        float length = getLength(segment);
        if (length > thres_length)
        {
            for (auto& pt : segment)
            {
                pt.label = segment_id;
                (*m_outcloud).push_back(pt);
            }
            segment_id += 1;
        }

        // remove points
        removePointsCloseToStrand(segment, is_removed, thres_thick);
    }
    printf("\n");

    genStrandsFromPointCloud(m_strands, m_outcloud);

    printf("Total number of setments: %d\n", segment_id);
}

float Hair::Gaussian(float x, float sigma)
{
    return exp(-(x * x) / (2 * sigma * sigma));
}

float Hair::getLength(const Strand & strand)
{
    float length = 0.f;

    int strand_size = (int)strand.size();

    if (strand_size < 2)
        return 0.f;

    for (int i = 1; i < strand_size; ++i)
    {
        const PointT& pt1 = strand[i - 1];
        const PointT& pt2 = strand[i];

        length += pcl::geometry::distance(pt1, pt2);
    }

    return length;
}

void Hair::performMeanShift(const std::vector<int>&k_indices, const PointT & pt, float sigma_e, float sigma_o, PointT & pt_new)
{
    // find intersection points of the plane and neighboring lines
    std::vector<PointT> intersect_pts;
    for (auto ind : k_indices)
    {
        PointT& tmp_pt = (*m_incloud)[ind];
        PointT intersect_pt;
        if (linePlaneIntersect(tmp_pt, pt, intersect_pt) > 0)
            intersect_pts.push_back(intersect_pt);
    }

    // flip normals
    for (auto& tmp_pt : intersect_pts)
    {
        if (pt.getNormalVector3fMap().dot(tmp_pt.getNormalVector3fMap()) < 0)
            tmp_pt.getNormalVector3fMap() *= -1;
    }

    // bilateral weights
    Eigen::VectorXd w_e(intersect_pts.size()); // Euclidean distance
    Eigen::VectorXd w_o(intersect_pts.size()); // Orientation distance
    for (int i = 0; i < intersect_pts.size(); ++i)
    {
        double dist_e = pcl::geometry::distance(pt, intersect_pts[i]);
        double diff_o = orientDifference(pt, intersect_pts[i]);
        w_e[i] = Gaussian(dist_e, sigma_e);
        w_o[i] = Gaussian(diff_o, sigma_o);
    }

    // get position of the new line
    pt_new = PointT();
    float w_sum = 0;
    for (int i = 0; i < intersect_pts.size(); ++i)
    {
        float w = w_e[i] * w_o[i];
        pt_new.getVector3fMap() += w * intersect_pts[i].getVector3fMap();
        pt_new.getNormalVector3fMap() += w * intersect_pts[i].getNormalVector3fMap();
        w_sum += w;
    }
    pt_new.getVector3fMap() /= w_sum;
    pt_new.getNormalVector3fMap() /= w_sum;
    pt_new.getNormalVector3fMap().normalize();
}

float Hair::orientDifference(const PointT & p1, const PointT & p2)
{
    Eigen::Vector3f n1 = p1.getNormalVector3fMap();
    Eigen::Vector3f n2 = p2.getNormalVector3fMap();
    return orientDifferenceImpl(n1, n2);
}

float Hair::orientDifferenceImpl(const Eigen::Vector3f & v1, const Eigen::Vector3f & v2)
{
    Eigen::Vector3f n1 = v1.normalized();
    Eigen::Vector3f n2 = v2.normalized();

    float tmpdot = fabsf(n1.dot(n2));

    tmpdot = std::max(std::min(tmpdot, 1.f), 0.f);

    float rad = acos(tmpdot);
    float deg = rad2deg(rad);

    return deg;
}

void Hair::removeZeroVertices(pcl::PointCloud<PointT>::Ptr cloud)
{
    pcl::PointCloud<PointT> pctmp;
    for (int i = 0; i < (*cloud).size(); i++)
    {
        PointT& pt = (*cloud)[i];
        if (!pt.getVector3fMap().isZero(0.f))
            pctmp.push_back(pt);
    }
    cloud->clear();
    for (auto& pt : pctmp)
        cloud->push_back(pt);
}

int Hair::linePlaneIntersect(PointT line, PointT pt, PointT & intersect_pt)
{
    float eps = 1e-16;

    Eigen::Vector3f V0 = pt.getVector3fMap();
    Eigen::Vector3f n = pt.getNormalVector3fMap();

    Eigen::Vector3f P0 = line.getVector3fMap();
    Eigen::Vector3f u = line.getNormalVector3fMap();

    float nu = n.dot(u);
    if (fabsf(nu) < eps)
        return -1;

    Eigen::Vector3f w = P0 - V0;
    float s1 = n.dot(-w) / nu;

    intersect_pt.getVector3fMap() = V0 + w + s1 * u;
    intersect_pt.getNormalVector3fMap() = u;

    return 1;
}

float Hair::rad2deg(float rad)
{
    return rad / M_PI * 180.f;
}

void Hair::saveStrands2Outcloud()
{
    m_outcloud->clear();

    for (auto& strand : m_strands)
        for (auto& pt : strand)
            m_outcloud->push_back(pt);
}

void Hair::mergeSegments(Strand & segment1, Strand & segment2, Strand & segment)
{
    for (int i = 0; i < segment1.size(); ++i)
        segment.push_back(segment1[segment1.size() - i - 1]);

    for (int i = 1; i < segment2.size(); ++i)
    {
        segment2[i].getNormalVector3fMap() = -segment2[i].getNormalVector3fMap();
        segment.push_back(segment2[i]);
    }
}

bool Hair::forwardEulerStep(const PointT & pt_now, const Eigen::Vector3f & direction, float nei_radius, float step_size, float thres_length, float thres_orient, const std::vector<bool>&is_removed, PointT & pt_next)
{
    bool found = false;
    PointT pt_tmp = PointT();

    if (std::isnan(direction[0]) || std::isnan(direction[1]) || std::isnan(direction[2]))
        return found;   // remove nan value to avoid following error in radiusSearch

    for (float move_step = step_size; move_step < thres_length; move_step += step_size)
    {
        pt_tmp.getVector3fMap() = pt_now.getVector3fMap() + (move_step * direction);

        // Neighbors containers
        std::vector<int> k_indices;
        std::vector<float> k_distances;
        m_tree->radiusSearch(pt_tmp, nei_radius, k_indices, k_distances);

        pt_tmp = PointT();

        int cnt_valid = 0;
        for (auto& ind : k_indices)
        {
            if (is_removed[ind])
                continue;

            Eigen::Vector3f pos = (*m_incloud)[ind].getVector3fMap();
            Eigen::Vector3f ori = (*m_incloud)[ind].getNormalVector3fMap().normalized();

            if (direction.dot(ori) < 0)
                ori = -ori;

            float angle = acosf(std::max(std::min(direction.dot(ori), 1.0f), 0.0f)) / M_PI * 180.f;
            float dist = (pt_now.getVector3fMap() - pos).norm();
            if (angle < thres_orient && dist > 1e-3)
            {
                cnt_valid++;
                pt_tmp.getVector3fMap() += pos;
                pt_tmp.getNormalVector3fMap() += ori;
            }
        }

        if (cnt_valid > 0)
        {
            pt_tmp.getVector3fMap() /= cnt_valid;
            pt_tmp.getNormalVector3fMap().normalize();

            if (move_step != step_size)
            {
                Eigen::Vector3f dir = (pt_tmp.getVector3fMap() - pt_now.getVector3fMap()).normalized();
                pt_next.getVector3fMap() = pt_now.getVector3fMap() + (step_size * dir);
                pt_next.getNormalVector3fMap() = dir;
            }
            else
                pt_next = pt_tmp;
            found = true;
            break;
        }
    }

    return found;
}

void Hair::removePointsCloseToStrand(const Strand & segment, std::vector<bool> &is_removed, float thres_thick)
{
    for (auto& pt : segment)
    {
        // Neighbors containers
        std::vector<int> k_indices;
        std::vector<float> k_distances;
        m_tree->radiusSearch(pt, 2 * thres_thick, k_indices, k_distances);

        // cylinder search
        std::vector<int> k_indices_tmp;
        Eigen::Vector4f line_pt = pt.getVector4fMap();
        Eigen::Vector4f line_dir = pt.getNormalVector4fMap();
        line_pt[3] = 0;
        for (auto& idx : k_indices)
        {
            Eigen::Vector4f tmp_pt = (*m_incloud)[idx].getVector4fMap();
            double pldist = sqrt(pcl::sqrPointToLineDistance(tmp_pt, line_pt, line_dir));
            if (pldist < thres_thick)
                k_indices_tmp.push_back(idx);
        }
        k_indices = k_indices_tmp;

        for (auto& ind : k_indices)
            is_removed[ind] = true;
    }
}

void Hair::genStrandsFromPointCloud(std::vector<Strand> &strands, pcl::PointCloud<PointT>::Ptr cloud)
{
    // gen strands next
    uint32_t id = 0;
    uint32_t id_current = (*cloud)[0].label;
    uint32_t id_next;

    for (auto& pt : *cloud)
    {
        id_next = pt.label;
        if (id_current != id_next)
        {
            id_current = id_next;
            id++;
        }
        pt.label = id;
    }
    uint32_t num_strands = id + 1; // zero-indexing

    // put strands
    strands.clear();
    strands.resize(num_strands);
    for (auto& s : strands)
        s.clear();

    for (auto& pt : *cloud)
        strands[pt.label].push_back(pt);
}

void Hair::writeOutcloud(std::string fn, bool binary)
{
    if (boost::algorithm::ends_with(fn, ".bin"))
    {
        std::ofstream out;
        out.open(fn, std::ios_base::binary);

        int num_strands = m_strands.size();
        int currentVertex = 0;

        out.write((char*)&num_strands, sizeof(int));
        std::cout << "Writing " << num_strands << " strands." << std::endl;

        for (unsigned int i = 0; i < num_strands; i++)
        {
            const Strand& tmpstrand = m_strands[i];
            int num_pts = tmpstrand.size();
            out.write((char*)&num_pts, sizeof(int));

            // For each strand, first read all of the vertices
            for (unsigned int j = 0; j < num_pts; j++)
            {
                const PointT& pt = tmpstrand[j];
                out.write((char*)&pt.x, sizeof(float));
                out.write((char*)&pt.y, sizeof(float));
                out.write((char*)&pt.z, sizeof(float));
                out.write((char*)&pt.normal_x, sizeof(float));
                out.write((char*)&pt.normal_y, sizeof(float));
                out.write((char*)&pt.normal_z, sizeof(float));
                out.write((char*)&pt.label, sizeof(uint32_t));
            }
        }
        out.close();
    }
    else if (boost::algorithm::ends_with(fn, ".ply"))
    {
        // if there is no outcloud, save incloud
        if (m_outcloud->size() == 0)
            *m_outcloud = *m_incloud;
        pcl::PLYWriter w;
        w.write<PointT>(fn, *m_outcloud, binary, false);
    }
    else
        std::cout << "Unsupported file type: " << fn << std::endl;
}

void Hair::writeOutcloudColor(std::string fn)
{
    // if there is no outcloud, save incloud with color
    if (m_outcloud->size() == 0)
        *m_outcloud = *m_incloud;

    pcl::PointCloud<PointColorT>::Ptr outcloud_color(new pcl::PointCloud<PointColorT>);

    auto prev_label = (*m_outcloud)[0].label;
    Eigen::Vector3i color = getColor(prev_label);

    for (auto& pt : (*m_outcloud))
    {
        PointColorT pt_c;

        pt_c.getVector3fMap() = pt.getVector3fMap();
        pt_c.getNormalVector3fMap() = pt.getNormalVector3fMap();

        if (prev_label != pt.label)
        {
            color = getColor(pt.label);
            prev_label = pt.label;
        }

        pt_c.r = color[0];
        pt_c.g = color[1];
        pt_c.b = color[2];

        outcloud_color->push_back(pt_c);
    }

    pcl::PLYWriter w;
    w.write<PointColorT>(fn, *outcloud_color, true, false);
}

Eigen::Vector3i Hair::getColor(int ind)
{
    srand(ind);
    uint8_t r = rand() % 256;
    uint8_t g = rand() % 256;
    uint8_t b = rand() % 256;
    return Eigen::Vector3i(r, g, b);
}

void Hair::computeStrandDirections()
{
    for (auto& strand : m_strands)
    {
        int pnumber = strand.size();
        for (int i = 0; i < pnumber; ++i)
        {
            PointT& pt = strand[i];

            if (i == 0)
            {
                PointT& pt_next = strand[i + 1];
                pt.getNormalVector3fMap() = (pt_next.getVector3fMap() - pt.getVector3fMap()).normalized();
            }
            else if (i == pnumber - 1)
            {
                PointT& pt_prev = strand[i - 1];
                pt.getNormalVector3fMap() = (pt.getVector3fMap() - pt_prev.getVector3fMap()).normalized();
            }
            else
            {
                PointT& pt_next = strand[i + 1];
                PointT& pt_prev = strand[i - 1];

                if (pt.label == pt_next.label && pt.label == pt_prev.label)
                    pt.getNormalVector3fMap() = (pt_next.getVector3fMap() - pt_prev.getVector3fMap()).normalized();
                else if (pt.label == pt_next.label && pt.label != pt_prev.label)
                    pt.getNormalVector3fMap() = (pt_next.getVector3fMap() - pt.getVector3fMap()).normalized();
                else if (pt.label != pt_next.label && pt.label == pt_prev.label)
                    pt.getNormalVector3fMap() = (pt.getVector3fMap() - pt_prev.getVector3fMap()).normalized();
                else
                    printf("something's wrong...\n");
            }
        }
    }
}

void Hair::loadRoots(std::string fn)
{
    m_roots_pos = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    m_fine_cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    m_poor_cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    m_conn_cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);

    m_roots_tree = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);
    m_fine_strands_tree = pcl::KdTreeFLANN<PointT>::Ptr(new pcl::KdTreeFLANN<PointT>);

    pcl::io::loadPLYFile(fn.c_str(), *m_roots_pos);
    m_roots_tree->setInputCloud(m_roots_pos);
}

Hair::Strand Hair::reverseStrand(Strand input_strand)
{
    Strand reversed_strand;
    for (auto iter = input_strand.rbegin(); iter != input_strand.rend(); iter++)
    {
        (*iter).normal_x = (*iter).normal_x * -1;
        (*iter).normal_y = (*iter).normal_y * -1;
        (*iter).normal_z = (*iter).normal_z * -1;
        reversed_strand.push_back(*iter);
    }
    return reversed_strand;
}

int Hair::strandNearest(Strand strand, PointT q_point)
{
    int idx = -1;
    float min_dis = 9999.f;
    for (int i_point = 0; i_point < strand.size(); i_point++)
    {
        float dis = pcl::euclideanDistance(q_point, strand[i_point]);
        if (dis < min_dis)
        {
            min_dis = dis;
            idx = i_point;
        }
    }
    return idx;
}

void Hair::getFinePoorStrands(int k_nearest, float thres_nn_roots_dis, float thres_length)
{
    float thres_angle_cos = std::cos(m_thres_angle_roots);

    int num_strands = (int)m_strands.size();
    printf("number of segments: %d\n", num_strands);
    
    std::vector<int> fine_segs_idx;
    std::vector<int> poor_segs_idx;
    
    std::vector<bool> fine_is_front;

    std::vector<PointT> fine_paired_roots;

    // Generate fine strands as guidance
    for (int i_strand = 0; i_strand < num_strands; i_strand++)
    {
        auto& strand = m_strands[i_strand];

        // get segments direction
        bool is_front = true;

        PointT seg_root = strand.front();
        PointT seg_tip = strand.back();
        
        pcl::Indices root_nn_idx;
        std::vector<float> root_nn_dis;
        m_roots_tree->nearestKSearch(seg_root, k_nearest, root_nn_idx, root_nn_dis);

        pcl::Indices tip_nn_idx;
        std::vector<float> tip_nn_dis;
        m_roots_tree->nearestKSearch(seg_tip, k_nearest, tip_nn_idx, tip_nn_dis);
        
        float avg_root_nn_dis = std::accumulate(root_nn_dis.begin(), root_nn_dis.end(), 0.f) / (float)k_nearest;
        float avg_tip_nn_dis = std::accumulate(tip_nn_dis.begin(), tip_nn_dis.end(), 0.f) / (float)k_nearest;

        if (avg_root_nn_dis >= thres_nn_roots_dis && avg_tip_nn_dis >= thres_nn_roots_dis)
        {
            poor_segs_idx.push_back(i_strand);
            continue;   // filter some outside segments / outlier, consider them as poor segments
        }

        is_front = avg_root_nn_dis < avg_tip_nn_dis ? true : false;
        
        if (!is_front)
        {
            seg_root = strand.back();
            seg_tip = strand.front();
            root_nn_idx = tip_nn_idx;
            root_nn_dis = tip_nn_dis;
        }
        
        // try to filter some support structure
        bool is_filter = true;
        for (int j_point = 0; j_point < strand.size(); j_point++)
        {
            PointT each_point = strand[j_point];
            pcl::Indices each_nn_idx;
            std::vector<float> each_nn_dis;
            m_roots_tree->nearestKSearch(each_point, 1, each_nn_idx, each_nn_dis);

            if (each_nn_dis[0] > thres_nn_roots_dis)
            {
                is_filter = false;
                break;
            }
        }

        if (is_filter)
            continue;

        // filter via orientation
        PointT& paired_root = (*m_roots_pos)[root_nn_idx[0]];
        const auto& paired_root_normal = paired_root.getNormalVector3fMap();
        const auto& seg_root_orien = seg_root.getNormalVector3fMap();

        if (paired_root_normal.dot(seg_root_orien) < thres_angle_cos && paired_root_normal.dot(seg_root_orien) < -1 * thres_angle_cos)
        {
            poor_segs_idx.push_back(i_strand);
            continue; // conside dis-connected segments as poor segmentsr
        }

        // TODO
        const auto& grow_dir = seg_root.getVector3fMap() - paired_root.getVector3fMap();
        auto grow_dir_normalized = grow_dir.normalized();
        if (paired_root_normal.dot(grow_dir_normalized) < 0)
        {
            poor_segs_idx.push_back(i_strand);
            continue; // conside dis-connected segments as poor segmentsr
        }
        
        // filter via length
        float length = getLength(strand);
        if (length < thres_length)
            continue;

        fine_segs_idx.push_back(i_strand);
        fine_is_front.push_back(is_front);

        fine_paired_roots.push_back(paired_root);
    }
    
    // get fine strands
    int num_fine_strands = fine_segs_idx.size();
    m_fine_strands.resize(num_fine_strands);
    for (int i_strand = 0; i_strand < num_fine_strands; i_strand++)
    {
        if (fine_is_front[i_strand])
        {
            m_fine_strands[i_strand] = m_strands[fine_segs_idx[i_strand]];
        }
        else    // correct the direction
        {
            Strand back_strand = m_strands[fine_segs_idx[i_strand]];
            Strand front_strand = reverseStrand(back_strand);
            m_fine_strands[i_strand] = front_strand;
        }

        PointT root = fine_paired_roots[i_strand];
        // root.label = i_strand;
        // (*m_fine_cloud).push_back(root);
        m_fine_strands[i_strand].push_front(root);
        for (auto& pt : m_fine_strands[i_strand])
        {
            pt.label = i_strand;
            (*m_fine_cloud).push_back(pt);
        }
    }

    // get poor strands
    int num_poor_strands = poor_segs_idx.size();
    m_poor_strands.resize(num_poor_strands);
    for (int i_strand = 0; i_strand < num_poor_strands; i_strand++)
    {
        m_poor_strands[i_strand] = m_strands[poor_segs_idx[i_strand]];
        for (auto& pt : m_poor_strands[i_strand])
        {
            pt.label = i_strand;
            (*m_poor_cloud).push_back(pt);
        }
    }

    printf("number of fine strands: %d, number of poor strands: %d\n", num_fine_strands, num_poor_strands);
}

bool Hair::poor2fineGenerationFrontBackward(Strand& poor_strand, Strand fine_strand, int nn_idx)
{
    PointT poor_front_pos = poor_strand[0];
    PointT trace_ver_pos = poor_front_pos;
    for (int i_v = nn_idx; i_v > 0; i_v--)
    {
        PointT p_0 = fine_strand[i_v - 1];
        PointT p_1 = fine_strand[i_v];
        PointT p_f = PointT();
        p_f.getVector3fMap() = trace_ver_pos.getVector3fMap() - (p_1.getVector3fMap() - p_0.getVector3fMap());
        p_f.label = poor_front_pos.label;
        p_f.getNormalVector3fMap() = p_1.getNormalVector3fMap();

        poor_strand.push_front(p_f);
        trace_ver_pos = p_f;

        // if can be connect to roots
        pcl::Indices each_nn_idx;
        std::vector<float> each_nn_dis;
        m_roots_tree->nearestKSearch(p_f, 1, each_nn_idx, each_nn_dis);
        if (each_nn_dis[0] <= m_thres_grow2root_dis)
            return true;
    }
    return false;
}

void Hair::poor2fineGenerationBackForward(Strand& poor_strand, Strand fine_strand, int nn_idx)
{
    PointT poor_front_pos = poor_strand[poor_strand.size() - 1];
    PointT trace_ver_pos = poor_front_pos;
    for (int i_v = nn_idx; i_v < fine_strand.size() - 1; i_v++)
    {
        PointT p_0 = fine_strand[i_v];
        PointT p_1 = fine_strand[i_v + 1];
        PointT p_f = PointT();
        p_f.getVector3fMap() = trace_ver_pos.getVector3fMap() + (p_1.getVector3fMap() - p_0.getVector3fMap());
        p_f.label = poor_front_pos.label;
        p_f.getNormalVector3fMap() = p_0.getNormalVector3fMap();

        poor_strand.push_back(p_f);
        trace_ver_pos = p_f;
    }
}

void Hair::connectPoorStrands()
{
    float poor_search_radius = m_poor_search_radius;
    float thres_nn_s_count = m_thres_nn_s_count;
    float thres_min_length = m_thres_min_length;
    float thres_angle_cos = std::cos(m_thres_angle_nn);

    int poor_search_count_max = 50; // for speeding up

    int num_fine_strands = m_fine_strands.size();
    int num_poor_strands = m_poor_strands.size();

    // build KD-Tree for fine strands points
    m_fine_strands_tree->setInputCloud(m_fine_cloud);

    for (int i_strand = 0; i_strand < num_poor_strands; i_strand++)
    {
        Strand poor_strand = m_poor_strands[i_strand];
        int num_strand_points = poor_strand.size();
        float strand_length = getLength(poor_strand);
        if (strand_length < thres_min_length)
            continue;

        std::map<int, int> nn_similarties;
        std::map<int, std::vector<float>> nn_distances;
        std::set<int> front_nn, back_nn;

        // TODO try to sample points instead of using all the points for saving time
        // for (auto& pt : m_poor_strands[i_strand])
        for(int j_point = 0; j_point < num_strand_points; j_point++)
        {
            auto& pt = poor_strand[j_point];
            pcl::Indices nn_fine_p_idx;
            std::vector<float> nn_fine_p_dis;
            // m_fine_strands_tree->radiusSearch(pt, poor_search_radius, nn_fine_p_idx, nn_fine_p_dis);
            m_fine_strands_tree->radiusSearch(pt, poor_search_radius, nn_fine_p_idx, nn_fine_p_dis, poor_search_count_max);
            for (int j_nn = 0; j_nn < nn_fine_p_idx.size(); j_nn++)
            {
                int nn_fine_s_idx = (*m_fine_cloud)[nn_fine_p_idx[j_nn]].label;
                // set nn for front and back
                if (j_point == 0)
                    front_nn.insert(nn_fine_s_idx);
                else if (j_point == num_strand_points - 1)
                    back_nn.insert(nn_fine_s_idx);

                float nn_p_dis = nn_fine_p_dis[j_nn];
                auto nn_fine_s_iter = nn_similarties.find(nn_fine_s_idx);
                if (nn_fine_s_iter == nn_similarties.end())
                {
                    nn_similarties.insert(std::make_pair(nn_fine_s_idx, 1));
                    std::vector<float> nn_fine_s_dis;
                    nn_distances.insert(std::make_pair(nn_fine_s_idx, nn_fine_s_dis));
                }
                else
                {
                    nn_fine_s_iter->second += 1;
                    // auto nn_fine_s_iter_dis = nn_distances.find(nn_fine_s_idx);
                    // nn_fine_s_iter_dis->second.push_back(nn_p_dis);
                    nn_distances[nn_fine_s_idx].push_back(nn_p_dis);
                }
            }
        }
        
        // front nn
        int max_sim_front = 0;
        int max_sim_front_idx = -1;
        for (auto nn_sim = nn_similarties.begin(); nn_sim != nn_similarties.end(); nn_sim++)
        {
            if (front_nn.find(nn_sim->first) != front_nn.end())
            {
                if (nn_sim->second > max_sim_front)
                {
                    max_sim_front = nn_sim->second;
                    max_sim_front_idx = nn_sim->first;
                }
            }
        }

        // back nn
        int max_sim_back = 0;
        int max_sim_back_idx = -1;
        for (auto nn_sim = nn_similarties.begin(); nn_sim != nn_similarties.end(); nn_sim++)
        {
            if (back_nn.find(nn_sim->first) != back_nn.end())
            {
                if (nn_sim->second > max_sim_back)
                {
                    max_sim_back = nn_sim->second;
                    max_sim_back_idx = nn_sim->first;
                }
            }
        }

        if (max_sim_front < thres_nn_s_count * poor_strand.size() 
            && max_sim_back < thres_nn_s_count * poor_strand.size())
            continue;

        if (max_sim_front_idx > -1 && max_sim_back_idx > -1)
        {
            // correct direction
            int nn_p_front_idx = strandNearest(m_fine_strands[max_sim_front_idx], poor_strand[0]);
            PointT nn_p_front = m_fine_strands[max_sim_front_idx][nn_p_front_idx];

            const auto& front_normal = poor_strand[0].getNormalVector3fMap();
            const auto& nn_front_p_normal = nn_p_front.getNormalVector3fMap();
            
            Strand front_fine_strand;
            Strand back_fine_strand;
            if (front_normal.dot(nn_front_p_normal) < 0) // wrong direction of the front
            {
                front_fine_strand = m_fine_strands[max_sim_back_idx];
                back_fine_strand = m_fine_strands[max_sim_front_idx];
                poor_strand = reverseStrand(poor_strand);

                nn_p_front_idx = strandNearest(front_fine_strand, poor_strand[0]);
                nn_p_front = front_fine_strand[nn_p_front_idx];
            }
            else
            {
                front_fine_strand = m_fine_strands[max_sim_front_idx];
                back_fine_strand = m_fine_strands[max_sim_back_idx];
            }

            const auto& temp_front_normal = poor_strand[0].getNormalVector3fMap();
            const auto& temp_front_nn_p_normal = nn_p_front.getNormalVector3fMap();
            if (temp_front_normal.dot(temp_front_nn_p_normal) < 0)
                continue;   // cannot connect because of the large normal distances
            // connect to front fine strand
            if(!poor2fineGenerationFrontBackward(poor_strand, front_fine_strand, nn_p_front_idx))
                continue;
            
            num_strand_points = poor_strand.size();
            int nn_p_back_idx = strandNearest(back_fine_strand, poor_strand[num_strand_points - 1]);
            PointT nn_p_back = back_fine_strand[nn_p_back_idx];
            const auto& temp_back_normal = poor_strand[num_strand_points - 1].getNormalVector3fMap();
            const auto& temp_back_nn_p_normal = nn_p_back.getNormalVector3fMap();
            if (temp_back_normal.dot(temp_back_nn_p_normal) < 0)
                continue;   // cannot connect because of the large normal distances
            // connect to back fine strand
            poor2fineGenerationBackForward(poor_strand, back_fine_strand, nn_p_back_idx);
            num_strand_points = poor_strand.size();
        }
        else if (max_sim_front_idx > -1 && max_sim_back_idx == -1)
        {   
            // backward grow at the front
            int nn_p_front_idx = strandNearest(m_fine_strands[max_sim_front_idx], poor_strand[0]);
            PointT nn_p_front = m_fine_strands[max_sim_front_idx][nn_p_front_idx];

            const auto& front_normal = poor_strand[0].getNormalVector3fMap();
            const auto& nn_front_p_normal = nn_p_front.getNormalVector3fMap();
            
            Strand front_fine_strand;
            if (front_normal.dot(nn_front_p_normal) < 0) // wrong direction of the front, drop it
                continue;
            else
                front_fine_strand = m_fine_strands[max_sim_front_idx];

            if(!poor2fineGenerationFrontBackward(poor_strand, front_fine_strand, nn_p_front_idx))
                continue;
            num_strand_points = poor_strand.size();
        }
        else if (max_sim_front_idx == -1 && max_sim_back_idx > -1)
        {
            // forward grow at the end
            int nn_p_back_idx = strandNearest(m_fine_strands[max_sim_back_idx], poor_strand[num_strand_points - 1]);
            PointT nn_p_back = m_fine_strands[max_sim_back_idx][nn_p_back_idx];

            const auto& back_normal = poor_strand[0].getNormalVector3fMap();
            const auto& nn_back_p_normal = nn_p_back.getNormalVector3fMap();
            
            Strand front_fine_strand;
            if (back_normal.dot(nn_back_p_normal) < 0)  // wrong direction of the end, drop it
            {
                front_fine_strand = m_fine_strands[max_sim_back_idx];
                poor_strand = reverseStrand(poor_strand);
            }
            else
                continue;

            if(!poor2fineGenerationFrontBackward(poor_strand, front_fine_strand, nn_p_back_idx))
                continue;
            num_strand_points = poor_strand.size();
        }

        for (auto& pt : poor_strand)
        {
            pt.label = i_strand;
            (*m_conn_cloud).push_back(pt);
        }
        m_conn_strands.push_back(poor_strand);
    }

    for (int i_strand = 0; i_strand < num_fine_strands; i_strand++)
    {
        for (auto& pt : m_fine_strands[i_strand])
        {
            pt.label = i_strand + num_poor_strands;
            (*m_conn_cloud).push_back(pt);
        }
        m_conn_strands.push_back(m_fine_strands[i_strand]);
    }
}

bool Hair::writeUSCData(std::string fn)
{
    FILE* f = fopen(fn.c_str(), "wb");
    if (!f)
    {
        fprintf(stderr, "Couldn't open %s\n", fn.c_str());
        return false;
    }

    int nstrands = m_conn_strands.size();
    fwrite(&nstrands, 4, 1, f);
    for (int i = 0; i < nstrands; i++)
    {
        int nverts = m_conn_strands[i].size();
        fwrite(&nverts, 4, 1, f);
        for (int j = 0; j < nverts; j++)
        {
            fwrite(&m_conn_strands[i][j].x, 4, 1, f);
            fwrite(&m_conn_strands[i][j].y, 4, 1, f);
            fwrite(&m_conn_strands[i][j].z, 4, 1, f);
        }
    }

    fclose(f);
    return true;
}

bool Hair::writeBin(std::string fn)
{
    FILE* f = fopen(fn.c_str(), "wb");
    if (!f)
    {
        fprintf(stderr, "Couldn't open %s\n", fn.c_str());
        return false;
    }

    int nstrands = m_conn_strands.size();
    fwrite(&nstrands, 4, 1, f);
    for (int i = 0; i < nstrands; i++)
    {
        int nverts = m_conn_strands[i].size();
        fwrite(&nverts, 4, 1, f);
        for (int j = 0; j < nverts; j++)
        {
            fwrite(&m_conn_strands[i][j].x, 4, 1, f);
            fwrite(&m_conn_strands[i][j].y, 4, 1, f);
            fwrite(&m_conn_strands[i][j].z, 4, 1, f);
            fwrite(&m_conn_strands[i][j].x, 4, 1, f);
            fwrite(&m_conn_strands[i][j].y, 4, 1, f);
            fwrite(&m_conn_strands[i][j].z, 4, 1, f);
            fwrite(&i, sizeof(uint32_t), 1, f);
        }
    }

    fclose(f);
    return true;
}