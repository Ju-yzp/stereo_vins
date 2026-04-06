cmake_minimum_required(VERSION 3.24)
project(
  stereo_vins
  VERSION 1.0.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(depthai CONFIG REQUIRED)
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(yaml-cpp 0.7.0 REQUIRED)

if(NOT TARGET pangolin::pangolin)
  add_library(pangolin::pangolin INTERFACE IMPORTED)

  if(TARGET pangolin)
    set(PANGOLIN_LINK_TARGET pangolin)
  else()
    set(PANGOLIN_LINK_TARGET ${Pangolin_LIBRARIES})
  endif()

  set_target_properties(
    pangolin::pangolin
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${Pangolin_INCLUDE_DIRS}"
               INTERFACE_LINK_LIBRARIES
               "${PANGOLIN_LINK_TARGET};GLEW::GLEW;OpenGL::GL")
endif()

add_library(core src/featureTracker.cpp src/estimator.cpp
                 src/featureManager.cpp src/visual_factor.cpp)

target_include_directories(core PUBLIC include)

target_link_libraries(
  core
  PUBLIC Eigen3::Eigen
         ${OpenCV_LIBS}
         ${GLEW_LIBRARIES}
         ${OpenGL_LIBRARIES}
         Ceres::ceres
         yaml-cpp
         depthai::core
         pangolin::pangolin)

add_executable(feature_tracker test/test_featureTracker.cpp
                               src/featureTracker.cpp)

target_include_directories(feature_tracker PRIVATE include)
target_link_libraries(
  feature_tracker PRIVATE depthai::core pangolin::pangolin yaml-cpp
                          ${OpenCV_LIBS} Eigen3::Eigen)
add_executable(estimator test/test_estimator.cpp)
target_link_libraries(estimator PRIVATE core)

add_executable(synchoronizer test/test_synchronizer.cpp)
target_include_directories(synchoronizer PRIVATE include)
target_link_libraries(
  synchoronizer PRIVATE depthai::core pangolin::pangolin yaml-cpp
                        ${OpenCV_LIBS} Eigen3::Eigen)

add_executable(params_parser test/test_paramsParser.cpp)
target_include_directories(params_parser PRIVATE include)
target_link_libraries(params_parser PRIVATE yaml-cpp ${OpenCV_LIBS}
                                            Eigen3::Eigen)
# target_link_libraries(syncBuffer PRIVATE core depthai::core
# pangolin::pangolin) if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
# CMAKE_CXX_COMPILER_ID STREQUAL "Clang") target_compile_options(core PUBLIC
# $<$<CONFIG:Debug>:-g -O0> $<$<CONFIG:Release>:-O3 -march=native>) endif()
