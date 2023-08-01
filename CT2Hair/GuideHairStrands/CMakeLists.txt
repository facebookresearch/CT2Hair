cmake_minimum_required(VERSION 3.2)
project("GuideHairStrands")

set(CMAKE_BUILD_TYPE Release)

if(UNIX)
    set(CUDAPATH /usr/local/cuda-11.5)
    set(OPENVDBPATH /home/shen/3rdparty/openvdb)
    set(OV_INCLUDE_DIR ${OPENVDBPATH}/build/openvdb/openvdb)
    set(OV_openvdb_INCLUDE_DIR ${OPENVDBPATH}/openvdb)
    set(OV_openvdb_LIBRARY ${OPENVDBPATH}/build/openvdb/openvdb/libopenvdb.so)

    find_package(Eigen3 REQUIRED)
    find_package(PCL 1.12 REQUIRED)

    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
elseif(WIN32)
    set(CUDAPATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5")
    set(VCPKGPATH D:/externalLibs/vcpkg)
    set(OPENVDBPATH D:/externalLibs/openvdb)
    set(OV_INCLUDE_DIR ${OPENVDBPATH}/build/openvdb/openvdb)
    set(OV_openvdb_INCLUDE_DIR ${OPENVDBPATH}/openvdb)
    set(OV_openvdb_LIBRARY ${OPENVDBPATH}/build/openvdb/openvdb/Release/openvdb.lib ${OPENVDBPATH}/build/openvdb/openvdb/Release/libopenvdb.lib)
    find_package(Eigen3 CONFIG REQUIRED)
    find_package(PCL CONFIG REQUIRED)
    find_package(flann CONFIG REQUIRED)
    find_package(OpenMP REQUIRED)
    include_directories(${VCPKGPATH}/installed/x64-windows/include)
    link_directories(${VCPKGPATH}/installed/x64-windows/lib)
endif()

enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -mtune=native -mavx -pthread")

set(SOURCES
    voxel_grid.cpp
    density_volume.cpp
    hair_vdb.cpp
    hair.cpp
    kernel.cu)

add_executable(GuideHairStrands main.cpp ${SOURCES})
target_include_directories(GuideHairStrands PRIVATE
                           ${CUDAPATH}/include
                           ${OV_INCLUDE_DIR}
                           ${OV_INCLUDE_DIR}/openvdb
                           ${OV_openvdb_INCLUDE_DIR})

if(UNIX)
    target_link_libraries(GuideHairStrands PRIVATE Eigen3::Eigen ${PCL_LIBRARIES} ${FLANN_LIBRARY} ${OV_openvdb_LIBRARY})
elseif(WIN32)
    target_link_libraries(GuideHairStrands PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen ${PCL_LIBRARIES} flann::flann flann::flann_cpp ${OV_openvdb_LIBRARY})
    set_property(TARGET GuideHairStrands PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    include(GNUInstallDirs)
    install(DIRECTORY ${OPENVDBPATH}/build/openvdb/openvdb/Release/ DESTINATION ${PROJECT_SOURCE_DIR}/Release FILES_MATCHING PATTERN "*.dll")
endif()
target_compile_features(GuideHairStrands PRIVATE cxx_std_17)
set_target_properties(GuideHairStrands PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(GuideHairStrands PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}")
