project(SPEAKER_ID)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -Ofast -g") #Change -O3 to -Ofast to run faster
enable_testing()

# look for Boost
if(DEFINED ENV{BOOST_ROOT})
  set(Boost_NO_SYSTEM_PATHS ON)
endif()
set(Boost_REALPATH ON)
find_package(Boost COMPONENTS program_options serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

# look for DyNet
# find_package(dynet REQUIRED)
# include_directories(${dynet_INCLUDE_DIR})

# look for Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# look for Threads
FIND_PACKAGE(Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

# CQT
include_directories("./cq-v1.1")

add_executable(speaker_id speaker_id.cc)
target_link_libraries(speaker_id dynet ${LIBS})
enable_testing()
