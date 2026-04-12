cmake_minimum_required(VERSION 3.15)
project(GradientCore VERSION 1.0.0 LANGUAGES CXX)

# Set default build type to Release for production
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect all source files recursively
file(GLOB_RECURSE SOURCES
    "src/**/*.cpp"
)

# Exclude platform_linux.cpp since it's included in platform.cpp
list(FILTER SOURCES EXCLUDE REGEX "platform_linux\\.cpp$")

# Verify source files were found
list(LENGTH SOURCES SOURCE_COUNT)
if(SOURCE_COUNT EQUAL 0)
    message(FATAL_ERROR "No source files found! Check your src/ directory structure")
else()
    message(STATUS "Found ${SOURCE_COUNT} source files")
endif()

# Create library target
add_library(gradientcore SHARED ${SOURCES})

# Add include directories
target_include_directories(gradientcore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Apply optimization flags to target (not global)
target_compile_options(gradientcore PRIVATE -O3 -mavx2 -mfma)

# Set version and SOVERSION for proper shared library versioning on Linux
set_target_properties(gradientcore PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
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

# Create directory for CMake config files
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/GradientCore")

# Create GradientCoreTargets.cmake file for install-tree
file(WRITE "${CMAKE_BINARY_DIR}/GradientCoreTargets.cmake"
"# Auto-generated targets file
include_guard()
add_library(GradientCore::gradientcore SHARED IMPORTED)
set_target_properties(GradientCore::gradientcore PROPERTIES
    IMPORTED_LOCATION \"\${_IMPORT_PREFIX}/lib/libgradientcore.so\"
    INTERFACE_INCLUDE_DIRECTORIES \"\${_IMPORT_PREFIX}/include\"
)
if(UNIX AND NOT APPLE)
    set_target_properties(GradientCore::gradientcore PROPERTIES
        IMPORTED_NO_SONAME ON
    )
endif()
# Link OpenMP
find_dependency(OpenMP REQUIRED)
target_link_libraries(GradientCore::gradientcore INTERFACE OpenMP::OpenMP_CXX)
")

# Install the targets file
install(FILES
    "${CMAKE_BINARY_DIR}/GradientCoreTargets.cmake"
    DESTINATION lib/cmake/GradientCore
)

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

# Handle version checking
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/GradientCoreConfigVersion.cmake"
    VERSION 1.0.0
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_BINARY_DIR}/GradientCoreConfigVersion.cmake"
    DESTINATION lib/cmake/GradientCore
)
