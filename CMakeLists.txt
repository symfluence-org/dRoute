cmake_minimum_required(VERSION 3.14)
project(dmc_route 
    VERSION 0.5.1 
    DESCRIPTION "Differentiable Muskingum-Cunge Routing"
    LANGUAGES CXX
)

# ============ Options ============
option(DMC_ENABLE_AD "Enable automatic differentiation via CoDiPack" ON)
option(DMC_ENABLE_ENZYME "Enable Enzyme AD (requires Clang with Enzyme plugin)" OFF)
option(DMC_ENABLE_NETCDF "Enable NetCDF input support" ON)
option(DMC_BUILD_TESTS "Build unit tests" ON)
option(DMC_BUILD_PYTHON "Build Python bindings via pybind11" OFF)
option(DMC_BUILD_SHARED "Build shared library" ON)

# ============ C++ Standard ============
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============ Dependencies via FetchContent ============
include(FetchContent)

# Use shallow clones to speed up downloads
set(FETCHCONTENT_QUIET OFF)

# CoDiPack (header-only AD library)
if(DMC_ENABLE_AD)
    FetchContent_Declare(
        codipack
        GIT_REPOSITORY https://github.com/SciCompKL/CoDiPack.git
        GIT_TAG v2.2.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(codipack)
    message(STATUS "CoDiPack enabled for automatic differentiation")
endif()

# Enzyme AD support
if(DMC_ENABLE_ENZYME)
    # Enzyme is a compiler PLUGIN, not a linked library
    # It transforms LLVM IR at compile time to generate derivatives
    
    # Try to find Enzyme plugin
    set(ENZYME_SEARCH_PATHS
        "/opt/homebrew/lib"
        "/opt/homebrew/Cellar/enzyme"
        "/usr/local/lib"
        "/usr/lib"
        "${CMAKE_PREFIX_PATH}/lib"
    )
    
    # Find the ClangEnzyme plugin
    find_file(ENZYME_PLUGIN
        NAMES 
            ClangEnzyme-21.dylib
            ClangEnzyme-20.dylib
            ClangEnzyme-19.dylib
            ClangEnzyme-18.dylib
            ClangEnzyme-17.dylib
            ClangEnzyme.dylib
            LLDEnzyme-21.dylib
            LLVMEnzyme-21.dylib
            ClangEnzyme-21.so
            ClangEnzyme-20.so
            ClangEnzyme-19.so
            ClangEnzyme.so
        PATHS ${ENZYME_SEARCH_PATHS}
        PATH_SUFFIXES enzyme
        NO_DEFAULT_PATH
    )
    
    # Also try globbing for any version
    if(NOT ENZYME_PLUGIN)
        file(GLOB ENZYME_PLUGIN_GLOB 
            "/opt/homebrew/lib/ClangEnzyme*.dylib"
            "/opt/homebrew/Cellar/enzyme/*/lib/ClangEnzyme*.dylib"
            "/usr/lib/ClangEnzyme*.so"
            "/usr/local/lib/ClangEnzyme*.so"
        )
        if(ENZYME_PLUGIN_GLOB)
            list(GET ENZYME_PLUGIN_GLOB 0 ENZYME_PLUGIN)
        endif()
    endif()
    
    if(ENZYME_PLUGIN)
        message(STATUS "Found Enzyme plugin: ${ENZYME_PLUGIN}")
        
        # Enzyme is used as a compiler plugin via -fplugin flag
        set(ENZYME_COMPILE_FLAGS "-fplugin=${ENZYME_PLUGIN}")
        set(DMC_ENZYME_AVAILABLE TRUE)
        
        # Check if using Clang
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message(WARNING "Enzyme requires Clang compiler. Current: ${CMAKE_CXX_COMPILER_ID}")
            message(WARNING "Enzyme AD will be available but may not work correctly.")
        endif()
        
    else()
        message(WARNING "")
        message(WARNING "==========================================================")
        message(WARNING "Enzyme plugin not found!")
        message(WARNING "")
        message(WARNING "Enzyme AD will be DISABLED.")
        message(WARNING "")
        message(WARNING "To enable Enzyme:")
        message(WARNING "  1. Install Enzyme: brew install enzyme  (macOS)")
        message(WARNING "  2. Or build from source: https://enzyme.mit.edu")
        message(WARNING "  3. Specify plugin path: -DENZYME_PLUGIN=/path/to/ClangEnzyme.so")
        message(WARNING "==========================================================")
        message(WARNING "")
        set(DMC_ENABLE_ENZYME OFF)
        set(DMC_ENZYME_AVAILABLE FALSE)
    endif()
endif()

# OpenMP for parallel routing
option(DMC_ENABLE_OPENMP "Enable OpenMP parallelization" OFF)
if(DMC_ENABLE_OPENMP)
    # On macOS with AppleClang, we need to help CMake find libomp from Homebrew
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(HOMEBREW_LIBOMP_PREFIX)
            message(STATUS "Found Homebrew libomp at: ${HOMEBREW_LIBOMP_PREFIX}")
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
            set(OpenMP_C_LIB_NAMES "omp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
        else()
            message(FATAL_ERROR "OpenMP requested but libomp not found. Install with: brew install libomp")
        endif()
    endif()
    
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP enabled for parallel routing")
    
    # CRITICAL: Enable thread-safe CoDiPack when using OpenMP
    # CoDiPack v2.2+ uses thread-local tapes by default, but we add
    # explicit defines for safety and compatibility
    add_compile_definitions(CODI_EnableOpenMP)
endif()

# BMI C++ specification (header-only, just download it)
set(BMI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/bmi")
if(NOT EXISTS "${BMI_INCLUDE_DIR}/bmi.hxx")
    message(STATUS "Downloading BMI C++ header...")
    file(MAKE_DIRECTORY ${BMI_INCLUDE_DIR})
    file(DOWNLOAD
        "https://raw.githubusercontent.com/csdms/bmi-cxx/master/bmi.hxx"
        "${BMI_INCLUDE_DIR}/bmi.hxx"
        STATUS download_status
    )
    list(GET download_status 0 status_code)
    if(NOT status_code EQUAL 0)
        message(FATAL_ERROR "Failed to download bmi.hxx")
    endif()
endif()

# NetCDF for reading model outputs
if(DMC_ENABLE_NETCDF)
    # Try to find system netCDF-cxx4
    find_package(netCDFCxx QUIET)
    if(NOT netCDFCxx_FOUND)
        # Fall back to netCDF C library with our own C++ wrapper
        find_package(netCDF QUIET)
        if(netCDF_FOUND)
            message(STATUS "Found netCDF C library, using thin C++ wrapper")
            set(DMC_USE_NETCDF_C ON)
        else()
            message(WARNING "NetCDF not found. Install libnetcdf-dev and libnetcdf-c++4-dev, or disable with -DDMC_ENABLE_NETCDF=OFF")
            set(DMC_ENABLE_NETCDF OFF)
        endif()
    else()
        message(STATUS "Found netCDF-cxx4")
        set(DMC_USE_NETCDF_CXX4 ON)
    endif()
endif()

# nlohmann/json for GeoJSON parsing
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

# ============ Main Library ============
add_library(dmc_route INTERFACE)

target_include_directories(dmc_route INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${BMI_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

if(DMC_ENABLE_AD)
    target_include_directories(dmc_route INTERFACE
        ${codipack_SOURCE_DIR}/include
    )
    target_compile_definitions(dmc_route INTERFACE DMC_USE_CODIPACK)
endif()

# Enzyme support
if(DMC_ENABLE_ENZYME AND DMC_ENZYME_AVAILABLE)
    target_compile_definitions(dmc_route INTERFACE DMC_USE_ENZYME)
    target_compile_options(dmc_route INTERFACE ${ENZYME_COMPILE_FLAGS})
    message(STATUS "Enzyme AD enabled with plugin: ${ENZYME_PLUGIN}")
endif()

# NetCDF support
if(DMC_ENABLE_NETCDF)
    target_compile_definitions(dmc_route INTERFACE DMC_USE_NETCDF)
    if(DMC_USE_NETCDF_CXX4)
        target_link_libraries(dmc_route INTERFACE netCDF::netcdf-cxx4)
        target_compile_definitions(dmc_route INTERFACE DMC_USE_NETCDF_CXX4)
    elseif(DMC_USE_NETCDF_C)
        target_link_libraries(dmc_route INTERFACE netCDF::netcdf)
        target_compile_definitions(dmc_route INTERFACE DMC_USE_NETCDF_C)
    endif()
endif()

# JSON for GeoJSON parsing
target_link_libraries(dmc_route INTERFACE nlohmann_json::nlohmann_json)

# Homebrew LLVM on macOS ships its own libc++ that matches the compiler headers.
# When using that toolchain, prefer its lib directory to avoid std:: symbols missing at link.
set(DMC_LLVM_LIBDIR "")
if(APPLE)
    if(DEFINED ENV{LLVM_PREFIX})
        set(DMC_LLVM_LIBDIR "$ENV{LLVM_PREFIX}/lib")
    else()
        get_filename_component(_dmc_clang_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(_dmc_llvm_prefix "${_dmc_clang_bin}" DIRECTORY)
        if(EXISTS "${_dmc_llvm_prefix}/lib/libc++.dylib")
            set(DMC_LLVM_LIBDIR "${_dmc_llvm_prefix}/lib")
        endif()
    endif()
endif()
set(DMC_LLVM_LIBCXX "")
set(DMC_LLVM_LIBCXXABI "")
if(DMC_LLVM_LIBDIR)
    if(EXISTS "${DMC_LLVM_LIBDIR}/libc++.dylib")
        set(DMC_LLVM_LIBCXX "${DMC_LLVM_LIBDIR}/libc++.dylib")
    elseif(EXISTS "${DMC_LLVM_LIBDIR}/libc++.1.dylib")
        set(DMC_LLVM_LIBCXX "${DMC_LLVM_LIBDIR}/libc++.1.dylib")
    endif()
    if(EXISTS "${DMC_LLVM_LIBDIR}/libc++abi.dylib")
        set(DMC_LLVM_LIBCXXABI "${DMC_LLVM_LIBDIR}/libc++abi.dylib")
    elseif(EXISTS "${DMC_LLVM_LIBDIR}/libc++abi.1.dylib")
        set(DMC_LLVM_LIBCXXABI "${DMC_LLVM_LIBDIR}/libc++abi.1.dylib")
    endif()
endif()
if(APPLE)
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libc++.dylib
        OUTPUT_VARIABLE _dmc_libcxx_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_dmc_libcxx_path AND NOT _dmc_libcxx_path STREQUAL "libc++.dylib" AND EXISTS "${_dmc_libcxx_path}")
        get_filename_component(DMC_LLVM_LIBDIR "${_dmc_libcxx_path}" DIRECTORY)
        set(DMC_LLVM_LIBCXX "${_dmc_libcxx_path}")
    endif()
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libc++abi.dylib
        OUTPUT_VARIABLE _dmc_libcxxabi_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_dmc_libcxxabi_path AND NOT _dmc_libcxxabi_path STREQUAL "libc++abi.dylib" AND EXISTS "${_dmc_libcxxabi_path}")
        set(DMC_LLVM_LIBCXXABI "${_dmc_libcxxabi_path}")
    endif()
endif()

# ============ Shared Library Wrapper ============
if(DMC_BUILD_SHARED)
    add_library(dmc_route_shared SHARED
        src/lib.cpp
    )
    target_link_libraries(dmc_route_shared PUBLIC dmc_route)
    set_target_properties(dmc_route_shared PROPERTIES
        OUTPUT_NAME dmc_route
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
    if(DMC_LLVM_LIBDIR)
        target_link_directories(dmc_route_shared PRIVATE "${DMC_LLVM_LIBDIR}")
        target_link_options(dmc_route_shared PRIVATE "-Wl,-rpath,${DMC_LLVM_LIBDIR}")
    endif()
    if(DMC_LLVM_LIBCXX)
        target_link_libraries(dmc_route_shared PRIVATE "${DMC_LLVM_LIBCXX}")
    endif()
    if(DMC_LLVM_LIBCXXABI)
        target_link_libraries(dmc_route_shared PRIVATE "${DMC_LLVM_LIBCXXABI}")
    endif()
endif()

# ============ Python Bindings (pybind11) ============
option(DMC_BUILD_PYTHON "Build Python bindings" OFF)
if(DMC_BUILD_PYTHON)
    # Find pybind11
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found, fetching from GitHub...")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
    
    message(STATUS "Building Python bindings")

    # Read version from _version.py
    file(READ "${CMAKE_SOURCE_DIR}/python/droute/_version.py" VERSION_FILE_CONTENTS)
    string(REGEX MATCH "__version__ = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _ ${VERSION_FILE_CONTENTS})
    set(DMC_VERSION ${CMAKE_MATCH_1})
    message(STATUS "dRoute version: ${DMC_VERSION}")

    # Create the Python module
    pybind11_add_module(_droute_core python/bindings.cpp)
    target_link_libraries(_droute_core PRIVATE dmc_route)
    target_compile_definitions(_droute_core PRIVATE DMC_VERSION="${DMC_VERSION}")
    if(DMC_LLVM_LIBDIR)
        target_link_directories(_droute_core PRIVATE "${DMC_LLVM_LIBDIR}")
        target_link_options(_droute_core PRIVATE "-Wl,-rpath,${DMC_LLVM_LIBDIR}")
    endif()
    if(DMC_LLVM_LIBCXX)
        target_link_libraries(_droute_core PRIVATE "${DMC_LLVM_LIBCXX}")
    endif()
    if(DMC_LLVM_LIBCXXABI)
        target_link_libraries(_droute_core PRIVATE "${DMC_LLVM_LIBCXXABI}")
    endif()
    
    # Set output name and properties
    if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set(_pydmc_route_output_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
    else()
        set(_pydmc_route_output_dir "${CMAKE_BINARY_DIR}/python")
    endif()
    set_target_properties(_droute_core PROPERTIES
        OUTPUT_NAME _droute_core
        LIBRARY_OUTPUT_DIRECTORY "${_pydmc_route_output_dir}"
    )
    
    # Install Python module
    if(DEFINED SKBUILD)
        # If building with scikit-build, install to the package directory
        install(TARGETS _droute_core DESTINATION .)
    else()
        # Otherwise install to lib/pythonX.Y/site-packages
        install(TARGETS _droute_core
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages
        )
    endif()

    message(STATUS "Python module will be built as: _droute_core")
endif()

if(DMC_ENABLE_ENZYME AND DMC_ENZYME_AVAILABLE AND DMC_BUILD_PYTHON AND TARGET _droute_core)
    target_compile_definitions(_droute_core PRIVATE DMC_USE_ENZYME)
    target_compile_options(_droute_core PRIVATE ${ENZYME_COMPILE_FLAGS})
endif()

# ============ Executable for Testing ============
option(DMC_BUILD_RUN "Build dmc_route_run CLI executable" ON)
if(DMC_BUILD_RUN)
    add_executable(dmc_route_run src/main.cpp)
    target_link_libraries(dmc_route_run PRIVATE dmc_route)
    if(APPLE)
        target_link_libraries(dmc_route_run PRIVATE c++)
    endif()
    if(DMC_LLVM_LIBDIR)
        target_link_directories(dmc_route_run PRIVATE "${DMC_LLVM_LIBDIR}")
        target_link_options(dmc_route_run PRIVATE "-Wl,-rpath,${DMC_LLVM_LIBDIR}")
    endif()
    if(DMC_LLVM_LIBCXX)
        target_link_libraries(dmc_route_run PRIVATE "${DMC_LLVM_LIBCXX}")
    endif()
    if(DMC_LLVM_LIBCXXABI)
        target_link_libraries(dmc_route_run PRIVATE "${DMC_LLVM_LIBCXXABI}")
    endif()
endif()

# ============ Tests ============
if(DMC_BUILD_TESTS)
    enable_testing()
    
    # Gradient verification test
    add_executable(test_gradients tests/test_gradients.cpp)
    target_link_libraries(test_gradients PRIVATE dmc_route)
    add_test(NAME gradient_verification COMMAND test_gradients)
    
    # Single reach routing test
    add_executable(test_single_reach tests/test_single_reach.cpp)
    target_link_libraries(test_single_reach PRIVATE dmc_route)
    add_test(NAME single_reach_routing COMMAND test_single_reach)
    
    # BMI interface test
    add_executable(test_bmi tests/test_bmi.cpp)
    target_link_libraries(test_bmi PRIVATE dmc_route)
    add_test(NAME bmi_interface COMMAND test_bmi)
    
    # Rigorous gradient verification suite (AD vs Finite Differences)
    add_executable(test_gradient_verification tests/test_gradient_verification.cpp)
    target_link_libraries(test_gradient_verification PRIVATE dmc_route)
    add_test(NAME gradient_verification_suite COMMAND test_gradient_verification)
    
    # Comprehensive test suite
    add_executable(test_comprehensive tests/test_comprehensive.cpp)
    target_link_libraries(test_comprehensive PRIVATE dmc_route)
    add_test(NAME comprehensive_suite COMMAND test_comprehensive)

    # Transfers, level-pool lakes, cycle rejection and mass conservation
    add_executable(test_transfers tests/test_transfers.cpp)
    target_link_libraries(test_transfers PRIVATE dmc_route)
    add_test(NAME transfers_and_lakes COMMAND test_transfers)

    # AD Backend Comparison test (CoDiPack vs Enzyme)
    add_executable(test_ad_backend_comparison tests/test_ad_backend_comparison.cpp)
    target_link_libraries(test_ad_backend_comparison PRIVATE dmc_route)
    if(DMC_ENABLE_ENZYME AND DMC_ENZYME_AVAILABLE)
        add_test(NAME ad_backend_comparison COMMAND test_ad_backend_comparison)
    endif()
endif()

# OpenMP linking for parallel routing
if(DMC_ENABLE_OPENMP)
    target_link_libraries(dmc_route INTERFACE OpenMP::OpenMP_CXX)
    if(DMC_BUILD_RUN)
        target_link_libraries(dmc_route_run PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(DMC_BUILD_TESTS)
        target_link_libraries(test_comprehensive PRIVATE OpenMP::OpenMP_CXX)
        target_link_libraries(test_ad_backend_comparison PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# Enzyme compile options for test targets
if(DMC_ENABLE_ENZYME AND DMC_ENZYME_AVAILABLE AND DMC_BUILD_TESTS)
    target_compile_options(test_ad_backend_comparison PRIVATE ${ENZYME_COMPILE_FLAGS})
endif()

# ============ Installation ============
include(GNUInstallDirs)

install(DIRECTORY include/dmc
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(DMC_BUILD_SHARED)
    install(TARGETS dmc_route_shared
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

if(DMC_BUILD_RUN)
    install(TARGETS dmc_route_run
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# ============ Package Configuration ============
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/dmc_routeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dmc_routeConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/dmc_routeConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dmc_route
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/dmc_routeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/dmc_routeConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dmc_route
)

# SUNDIALS support for Saint-Venant solver
option(DMC_ENABLE_SUNDIALS "Enable SUNDIALS for SVE solver" OFF)

if(DMC_ENABLE_SUNDIALS)
    if(DEFINED SUNDIALS_ROOT)
        set(SUNDIALS_DIR "${SUNDIALS_ROOT}/lib/cmake/sundials")
    endif()
    
    find_package(SUNDIALS REQUIRED COMPONENTS cvodes nvecserial sunlinsoldense sunmatrixdense)
    
    add_definitions(-DDMC_ENABLE_SUNDIALS)
    
    if(DMC_BUILD_PYTHON AND TARGET _droute_core)
        target_link_libraries(_droute_core PRIVATE
            SUNDIALS::cvodes
            SUNDIALS::nvecserial
            SUNDIALS::sunlinsoldense
            SUNDIALS::sunmatrixdense
        )
    endif()
    
    message(STATUS "SUNDIALS enabled for Saint-Venant solver")
endif()
