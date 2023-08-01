// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <iostream>

#include "clock.h"
#include "hair_vdb.h"
#include "hair.h"
#include "density_volume.h"

// Calculate orientations
void calculateOrientations(char* argv[])
{   
    /*
    * ./GuideHairStrands 0 ../../../data/VDBs/Clipped/em_mid_clipped.vdb 0.132319 ../../../data/PointClouds/em_mid/roots_rec.ply 1.8 temp.ply
    */
    int argcnt = 2;
    std::string vdb_name(argv[argcnt++]);
    float voxel_size = atof(argv[argcnt++]);
    std::string roots_name = argv[argcnt++];
    float wignet_dis_thres = atof(argv[argcnt++]);
    std::string output_name(argv[argcnt++]);

    float voxel_value_max = 1.0f;

    Clock clk;
    clk.print();
    HairVDB* hairvdb;

    clk.tick();
    hairvdb = new HairVDB(vdb_name, voxel_size, roots_name);
    hairvdb->m_wig_dis_thres = wignet_dis_thres;
    printf("loading takes %.2fs\n", clk.tock());

    // get orientation
    clk.tick();
    hairvdb->getPoints();
    printf("getting points takes %.2fs\n", clk.tock());
    clk.tick();
    hairvdb->calculateOriens();
    printf("calculating orientations takes %.2fs\n", clk.tock());
    hairvdb->releaseMem();
    clk.tick();
    // hairvdb->exportPts(output_name, false);
    hairvdb->exportPtsWithValue(output_name, voxel_value_max, false);
    printf("exporting points takes %.2fs\n", clk.tock());
}

// Guide hair strands generation
void genGuideStrands(char* argv[])
{
    /*
    * ./GuideHairStrands 1 temp.ply temp_filtered.ply 2.0 10 0.1 30 0.002 1000 1 0 0.36 0.36 20 6 ../../../data/PointClouds/em_mid/roots_rec.ply 3 40 5
    */
    int argcnt = 2;
    
    /* 
    * 1. Filter points using Meanshift
    */
    // parameters for orientation filtering
    std::string fn_incloud(argv[argcnt++]);
    std::string fn_outcloud(argv[argcnt++]);
    float nei_radius = atof(argv[argcnt++]);    // mm
    float sigma_e = atof(argv[argcnt++]);   // mm
    float sigma_o = atof(argv[argcnt++]);   // degree
    float thres_shift = atof(argv[argcnt++]);   // mm
    bool use_cuda = (1 == atoi(argv[argcnt++]));
    int gpu_id = atoi(argv[argcnt++]);
    int nei_thres = 10;
    int max_num_shift = 1000;

    Clock clk;
    clk.print();

    // load point cloud
    printf("Loading orientation point cloud...\n");
    clk.tick();
    Hair hair(fn_incloud, false, true);
    printf("Orientations loaded: %.2f seconds\n", clk.tock());

    // mean shift
    clk.tick();
    if (use_cuda)
        hair.meanShiftCUDA(nei_radius, nei_thres, sigma_e, sigma_o, thres_shift, max_num_shift, gpu_id);
    else
        hair.meanShift(nei_radius, nei_thres, sigma_e, sigma_o, thres_shift, max_num_shift);

    printf("MeanShift done: %.2f seconds\n", clk.tock());


    /* 
    * 2. Hair segments generation
    */
    // parameters for hair segments generation
    float nei_radius_seg = atof(argv[argcnt++]);    // mm
    float thres_orient = atof(argv[argcnt++]);  // deg
    float thres_length = atof(argv[argcnt++]);  // mm
    float step_size = nei_radius_seg;   // mm
    float thres_thick = nei_radius_seg; // mm

    hair.out2inCloud();

    // hair segment
    clk.tick();
    hair.genSegmentsFromPointCloud(nei_radius_seg, step_size, thres_orient, thres_length, thres_thick);
    printf("Hair segment: %.2f seconds\n", clk.tock());


    /* 
    * 3. Hair strands growing
    */
    // parameters for hair growing
    std::string fn_inroots(argv[argcnt++]);
    float thres_nn_roots_dis = atof(argv[argcnt++]);    // mm
    float thres_length_grow = atof(argv[argcnt++]);     // mm
    int k_nearest = 3;

    hair.out2inCloud(true);

    printf("Loading roots...\n");
    hair.loadRoots(fn_inroots);
    printf("Load done\n");

    // get fine
    clk.tick();
    hair.getFinePoorStrands(k_nearest, thres_nn_roots_dis, thres_length_grow);
    printf("Got fine hair strands: %.2f seconds\n", clk.tock());

    // grow poor
    clk.tick();
    hair.connectPoorStrands();
    printf("Growing done: %.2f seconds\n", clk.tock());

    // save connected outlier
    clk.tick();
    hair.m_outcloud = hair.m_conn_cloud;
    hair.writeOutcloudColor(fn_outcloud);
    printf("Save connected strands as point cloud: %.2f seconds\n", clk.tock());
    
    // write binary file
    clk.tick();
    hair.writeBin(fn_outcloud.substr(0, fn_outcloud.size() - 4) + ".bin");
    printf("Save connected strands as bin: %.2f seconds\n", clk.tock());
}

int main(int argc, char* argv[])
{
    int module_idx = atoi(argv[1]);
    if (module_idx == 0)
        calculateOrientations(argv);
    else if (module_idx == 1)
        genGuideStrands(argv);
    return 0;
}