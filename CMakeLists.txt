cmake_minimum_required(VERSION 3.24)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" FUSE_VERSION)
string(STRIP "${FUSE_VERSION}" FUSE_VERSION)
if(NOT FUSE_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR "VERSION must contain a release version MAJOR.MINOR.PATCH")
endif()
set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION"
)
project(
  FUSE
  VERSION ${FUSE_VERSION}
  LANGUAGES CXX CUDA
)

option(FUSE_BUILD_PYTHON "Build the fusemesh Python extension" ON)
option(FUSE_BUILD_TESTS "Build native validation tests" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Explicit fat binary: Ampere, Ada, Hopper plus forward PTX. Never host-only.
set(FUSE_CUDA_ARCHITECTURES
    "80-real;86-real;89-real;90-real;90-virtual"
    CACHE STRING "CUDA architectures"
)
set(CMAKE_CUDA_ARCHITECTURES "${FUSE_CUDA_ARCHITECTURES}")
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
find_package(CUDAToolkit 12.8 REQUIRED)
if(NOT CUDAToolkit_VERSION VERSION_LESS 12.9)
  message(FATAL_ERROR "FUSE currently supports the CUDA 12.8 toolkit family only")
endif()

set(FUSE_NANOVDB_REVISION "e26f5a569201d3236e0c30580572101022d3d754")
set(NANOVDB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/openvdb/nanovdb")
if(NOT EXISTS "${NANOVDB_ROOT}/nanovdb/NanoVDB.h")
  message(FATAL_ERROR "NanoVDB sources missing. Run git submodule update --init --recursive")
endif()
# Check a checkout when Git is available; self-contained sdists need no Git.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/openvdb/.git")
  find_package(Git QUIET)
  if(Git_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/openvdb"
      OUTPUT_VARIABLE revision
      OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY
    )
    if(NOT revision STREQUAL FUSE_NANOVDB_REVISION)
      message(FATAL_ERROR "NanoVDB checkout differs from the pinned revision")
    endif()
  endif()
endif()
string(REPLACE ";" "," FUSE_ARCH_STRING "${FUSE_CUDA_ARCHITECTURES}")
configure_file(cmake/build_config.h.in generated/fuse/build_config.h @ONLY)
add_library(
  fuse STATIC src/cuda/smoke.cu src/core/reconstruct.cpp src/core/version.cpp
              src/geometry/scene.cpp src/volume/topology.cpp src/cuda/reconstruct.cu
)
add_library(FUSE::fuse ALIAS fuse)
target_include_directories(
  fuse
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${NANOVDB_ROOT}"
          "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
target_link_libraries(fuse PUBLIC CUDA::cudart_static)
target_compile_options(fuse PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--fmad=false>)
set_target_properties(fuse PROPERTIES CUDA_RUNTIME_LIBRARY Static)

if(FUSE_BUILD_PYTHON)
  find_package(
    Python 3.9
    COMPONENTS Interpreter Development.Module
    REQUIRED
  )
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY
  )
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_native bindings/module.cpp)
  target_link_libraries(_native PRIVATE FUSE::fuse)
  install(TARGETS _native LIBRARY DESTINATION fusemesh)
endif()
if(NOT SKBUILD)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  install(
    TARGETS fuse
    EXPORT FUSETARGETS
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(
    EXPORT FUSETARGETS
    NAMESPACE FUSE::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FUSE
  )
  configure_package_config_file(
    cmake/FUSEConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/FUSEConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FUSE
  )
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/FUSEConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY ExactVersion
  )
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/FUSEConfigVersion.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/FUSEConfig.cmake"
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FUSE
  )
endif()
if(FUSE_BUILD_TESTS)
  enable_testing()
  add_executable(fuse_version tests/native_version.cpp)
  target_compile_definitions(fuse_version PRIVATE FUSE_EXPECTED_VERSION="${PROJECT_VERSION}")
  target_link_libraries(fuse_version PRIVATE FUSE::fuse)
  add_test(NAME fuse_version COMMAND fuse_version)
  add_executable(fuse_smoke tests/native_smoke.cpp)
  target_link_libraries(fuse_smoke PRIVATE FUSE::fuse)
  add_test(NAME fuse_smoke COMMAND fuse_smoke)
  add_executable(fuse_cells tests/native_cells.cpp)
  target_include_directories(fuse_cells PRIVATE src include)
  add_test(NAME fuse_cells COMMAND fuse_cells)
  add_executable(fuse_reconstruct tests/native_reconstruct.cpp)
  target_link_libraries(fuse_reconstruct PRIVATE FUSE::fuse)
  add_test(NAME fuse_reconstruct COMMAND fuse_reconstruct)
  add_executable(fuse_extraction tests/native_extraction.cpp)
  target_include_directories(fuse_extraction PRIVATE src)
  target_link_libraries(fuse_extraction PRIVATE FUSE::fuse)
  add_test(NAME fuse_extraction COMMAND fuse_extraction)
endif()
