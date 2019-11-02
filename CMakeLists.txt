cmake_minimum_required(VERSION 3.0.0)
project(QDTree)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
include_directories("${JSONCPP_INCLUDE_DIRS}")

include_directories(src)
set(SOURCES src/main.cpp src/qdtree-input.cpp src/qdtree-tests.cpp src/qdtree-const.cpp)

add_executable(tree_test ${SOURCES})
target_compile_options(tree_test PRIVATE -Werror -Wall -Wextra -g)
target_link_libraries(tree_test Boost::program_options)
target_link_libraries(tree_test jsoncpp)
