cmake_minimum_required(VERSION 3.15...3.27)
project(xray_epipolar_consistency LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Fetch Eigen (Header only, so it's simple to include this way for the scaffold)
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(eigen)

# Gather source files
file(GLOB_RECURSE SRC_FILES "src/*.cpp")
list(FILTER SRC_FILES EXCLUDE REGEX "Test.*\\.cpp")
list(FILTER SRC_FILES EXCLUDE REGEX ".*\\.ipynb_checkpoints.*")

file(GLOB_RECURSE CUDA_FILES "src/*.cu")
list(FILTER CUDA_FILES EXCLUDE REGEX "Test.*\\.cu")
list(FILTER CUDA_FILES EXCLUDE REGEX ".*\\.ipynb_checkpoints.*")
list(APPEND SRC_FILES ${CUDA_FILES})

# Create the python module
pybind11_add_module(_core ${SRC_FILES})

# Add include directories for the user's libraries
target_include_directories(_core PRIVATE
    src
    src/HeaderOnly
    src/GetSet
    src/LibEpipolarConsistency
    src/LibProjectiveGeometry
    src/LibUtilsCuda
    src/LibRayCastBackproject
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(_core PRIVATE USE_CUDA=1)
target_link_libraries(_core PRIVATE CUDA::cudart CUDA::cufft Eigen3::Eigen)

# Install the built extension into the python package directory
install(TARGETS _core DESTINATION xray_epipolar_consistency)
