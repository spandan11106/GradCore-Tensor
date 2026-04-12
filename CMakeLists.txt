cmake_minimum_required(VERSION 3.10)
project(GradientCore)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma")

# Collect all source files
file(GLOB_RECURSE SOURCES
    "src/autograd/*.cpp"
    "src/nn/*.cpp"
    "src/optim/*.cpp"
    "src/tensor/*.cpp"
)

# Remove platform_linux.cpp since it's included in platform.cpp
list(FILTER SOURCES EXCLUDE REGEX "platform_linux\\.cpp$")

# Create library target
add_library(gradientcore SHARED ${SOURCES})

# Add include directories
target_include_directories(gradientcore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Find and link OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(gradientcore PUBLIC OpenMP::OpenMP_CXX)

# Installation targets
install(TARGETS gradientcore
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY include/ DESTINATION include)

# Create GradientCoreConfig.cmake for find_package support
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/GradientCoreConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/GradientCoreConfig.cmake"
    @ONLY
)

install(FILES
    "${CMAKE_BINARY_DIR}/GradientCoreConfig.cmake"
    DESTINATION lib/cmake/GradientCore
)
