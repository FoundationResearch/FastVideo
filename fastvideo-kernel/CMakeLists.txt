cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(fastvideo-kernel LANGUAGES CXX)

# Prefer environment variable (used by CI or pip install git+repo_addr) if CMake var is not explicitly set.
if(NOT DEFINED GPU_BACKEND AND DEFINED ENV{GPU_BACKEND})
    set(GPU_BACKEND "$ENV{GPU_BACKEND}")
endif()

if(GPU_BACKEND STREQUAL "ROCM")
    enable_language(HIP)
else()
    enable_language(CUDA)
    # Ensure CUDA toolkit targets (CUDA::cudart, CUDA::cuda_driver, etc.) are available.
    find_package(CUDAToolkit REQUIRED)
endif()

# Import common utils if needed, but we keep it simple for now

# Find Python and Torch
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Robustly find Torch include paths using Python
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()))"
    OUTPUT_VARIABLE TORCH_INCLUDE_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_PATHS})

# Find Torch package (still useful for libraries)
find_package(Torch REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/cutlass/include
    ${CMAKE_SOURCE_DIR}/csrc
    ${CMAKE_SOURCE_DIR}/csrc/turbodiffusion
    ${TORCH_INCLUDE_DIRS}
)

# ------------------------------------
# ThunderKittens (TK) per-arch toggles
# ------------------------------------
# We support *two* TK versions side-by-side:
# - SM90/H100 kernels: include/tk (existing pinned TK)
# - SM100/B200 kernels: include/tk_b200 (newer TK needed by B200)
#
# IMPORTANT: do NOT add both TK include paths to the same compilation unit.
# We use per-arch object libraries to isolate include paths + arch flags.
#
# SM90 uses the existing FASTVIDEO_KERNEL_BUILD_TK switch (backward compatible).
# SM100 is controlled by FASTVIDEO_KERNEL_BUILD_TK_SM100.
#
# AUTO behavior:
# - SM90: enables if TORCH_CUDA_ARCH_LIST contains 9.0a/90a/sm_90a
# - SM100: enables if TORCH_CUDA_ARCH_LIST contains 10.0/100/sm_100
set(FASTVIDEO_KERNEL_BUILD_TK "AUTO" CACHE STRING "Build ThunderKittens SM90 kernels: AUTO/ON/OFF (backward compatible)")
set_property(CACHE FASTVIDEO_KERNEL_BUILD_TK PROPERTY STRINGS AUTO ON OFF)
set(FASTVIDEO_KERNEL_BUILD_TK_SM100 "OFF" CACHE STRING "Build ThunderKittens SM100 kernels (B200): AUTO/ON/OFF")
set_property(CACHE FASTVIDEO_KERNEL_BUILD_TK_SM100 PROPERTY STRINGS AUTO ON OFF)

# Prefer environment variable (used by CI) if CMake var is not explicitly set.
if(NOT DEFINED TORCH_CUDA_ARCH_LIST AND DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    set(TORCH_CUDA_ARCH_LIST "$ENV{TORCH_CUDA_ARCH_LIST}")
endif()

message(STATUS "TORCH_CUDA_ARCH_LIST (cmake/env): ${TORCH_CUDA_ARCH_LIST}")
message(STATUS "FASTVIDEO_KERNEL_BUILD_TK (SM90): ${FASTVIDEO_KERNEL_BUILD_TK}")
message(STATUS "FASTVIDEO_KERNEL_BUILD_TK_SM100: ${FASTVIDEO_KERNEL_BUILD_TK_SM100}")

set(ENABLE_TK_SM90 OFF)
set(ENABLE_TK_SM100 OFF)

# Helper: detect arch presence in TORCH_CUDA_ARCH_LIST
function(_fastvideo_has_arch _regex _outvar)
    set(${_outvar} OFF PARENT_SCOPE)
    if(TORCH_CUDA_ARCH_LIST)
        string(REGEX MATCH "${_regex}" _HAS "${TORCH_CUDA_ARCH_LIST}")
        if(_HAS)
            set(${_outvar} ON PARENT_SCOPE)
        endif()
    endif()
endfunction()

# ---- SM90 (H100 / sm_90a) ----
if(FASTVIDEO_KERNEL_BUILD_TK STREQUAL "ON")
    set(ENABLE_TK_SM90 ON)
elseif(FASTVIDEO_KERNEL_BUILD_TK STREQUAL "OFF")
    set(ENABLE_TK_SM90 OFF)
else()
    # AUTO: detect Hopper if possible.
    if(TORCH_CUDA_ARCH_LIST)
        # Accept common spellings: 9.0a, 90a, sm_90a.
        string(REGEX MATCH "(^|[; ,])((9\\.0a)|(90a)|(sm_90a))([; ,]|$)" _HAS_90A "${TORCH_CUDA_ARCH_LIST}")
        if(_HAS_90A)
            set(ENABLE_TK_SM90 ON)
        endif()
    else()
        # Best-effort local detection (works when a CUDA device is visible).
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import torch; import sys; \nprint('1' if (torch.cuda.is_available() and torch.version.cuda and torch.cuda.get_device_capability()[0] >= 9) else '0')"
            OUTPUT_VARIABLE _LOCAL_HAS_HOPPER
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_LOCAL_HAS_HOPPER STREQUAL "1")
            set(ENABLE_TK_SM90 ON)
        endif()
    endif()
endif()

# ---- SM100 (B200 / assume compute capability 10.0) ----
if(FASTVIDEO_KERNEL_BUILD_TK_SM100 STREQUAL "ON")
    set(ENABLE_TK_SM100 ON)
elseif(FASTVIDEO_KERNEL_BUILD_TK_SM100 STREQUAL "OFF")
    set(ENABLE_TK_SM100 OFF)
else()
    # Accept common spellings: 10.0a/100a/sm_100a (tcgen05) and 10.0/100/sm_100.
    _fastvideo_has_arch("(^|[; ,])((10\\.0a)|(100a)|(sm_100a)|(10\\.0)|(100)|(sm_100))([; ,]|$)" _HAS_100)
    if(_HAS_100)
        set(ENABLE_TK_SM100 ON)
    endif()
endif()

if(ENABLE_TK_SM90)
    message(STATUS "ThunderKittens SM90 kernels: ENABLED")
else()
    message(STATUS "ThunderKittens SM90 kernels: DISABLED (will use Triton fallbacks at runtime)")
endif()
if(ENABLE_TK_SM100)
    message(STATUS "ThunderKittens SM100 kernels: ENABLED")
else()
    message(STATUS "ThunderKittens SM100 kernels: DISABLED (will use Triton fallbacks at runtime)")
endif()

# Always try to build the extension if CUDA is available, but conditionally add sources/flags
set(BUILD_CXX_KERNELS ON)

# Compiler flags
set(CUDA_FLAGS
    "-DNDEBUG"
    "-O3"
    "-std=c++20"
    "--use_fast_math"
    "--expt-extended-lambda"
    "--expt-relaxed-constexpr"
    "-Xcompiler=-fno-strict-aliasing"
    "-Xcompiler=-fPIC"
    "-DTORCH_COMPILE"
    "-Xnvlink=--verbose"
    "-Xptxas=--verbose"
    "-Xptxas=--warn-on-spills"
)

message(STATUS "CMAKE_CUDA_ARCHITECTURES (global): ${CMAKE_CUDA_ARCHITECTURES}")

if(BUILD_CXX_KERNELS)
    # Source files
    set(EXTENSION_SOURCES 
        csrc/common_extension.cpp
        csrc/turbodiffusion/gemm/gemm.cu
        csrc/turbodiffusion/norm/rmsnorm.cu
        csrc/turbodiffusion/norm/layernorm.cu
        csrc/turbodiffusion/quant/quant.cu
    )
    
    # Combined FastVideo Extension
    # Using name 'fastvideo_kernel_ops' to distinguish from the python package namespace
    Python_add_library(fastvideo_kernel_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI
        ${EXTENSION_SOURCES}
    )

    # Build compile definitions list
    set(COMPILE_DEFS TORCH_EXTENSION_NAME=fastvideo_kernel_ops)
    target_compile_definitions(fastvideo_kernel_ops PRIVATE ${COMPILE_DEFS})

    target_compile_options(fastvideo_kernel_ops PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
    )

    # -------------------------------
    # TK kernels built as object libs
    # -------------------------------
    if(ENABLE_TK_SM90)
        if(NOT EXISTS "${CMAKE_SOURCE_DIR}/include/tk/include/kittens.cuh")
            message(FATAL_ERROR "SM90 TK enabled but include/tk submodule is missing. Run: git submodule update --init --recursive")
        endif()

        add_library(fastvideo_kernel_tk_sm90_objs OBJECT
            csrc/attention/st_attn_h100.cu
            csrc/attention/block_sparse_h100.cu
        )
        set_property(TARGET fastvideo_kernel_tk_sm90_objs PROPERTY CUDA_ARCHITECTURES "90a")
        target_include_directories(fastvideo_kernel_tk_sm90_objs PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/include/cutlass/include
            ${CMAKE_SOURCE_DIR}/include/tk/include
            ${CMAKE_SOURCE_DIR}/include/tk/prototype
            ${CMAKE_SOURCE_DIR}/csrc
            ${TORCH_INCLUDE_DIRS}
            ${Python_INCLUDE_DIRS}
        )
        target_compile_definitions(fastvideo_kernel_tk_sm90_objs PRIVATE
            TORCH_EXTENSION_NAME=fastvideo_kernel_ops
            KITTENS_HOPPER
            TK_COMPILE_ST_ATTN
            TK_COMPILE_BLOCK_SPARSE
        )
        target_compile_options(fastvideo_kernel_tk_sm90_objs PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
        )
        target_sources(fastvideo_kernel_ops PRIVATE $<TARGET_OBJECTS:fastvideo_kernel_tk_sm90_objs>)
        target_compile_definitions(fastvideo_kernel_ops PRIVATE TK_COMPILE_ST_ATTN TK_COMPILE_BLOCK_SPARSE)
    endif()

    if(ENABLE_TK_SM100)
        if(NOT EXISTS "${CMAKE_SOURCE_DIR}/include/tk_b200/include/kittens.cuh")
            message(FATAL_ERROR "SM100 TK enabled but include/tk_b200 submodule is missing. Create it as a separate ThunderKittens submodule at fastvideo-kernel/include/tk_b200 and run: git submodule update --init --recursive")
        endif()

        add_library(fastvideo_kernel_tk_sm100_objs OBJECT
            csrc/attention/block_sparse_b200.cu
        )
        # tcgen05 instructions require SM100a.
        set_property(TARGET fastvideo_kernel_tk_sm100_objs PROPERTY CUDA_ARCHITECTURES "100a")
        target_include_directories(fastvideo_kernel_tk_sm100_objs PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/include/cutlass/include
            ${CMAKE_SOURCE_DIR}/include/tk_b200/include
            ${CMAKE_SOURCE_DIR}/include/tk_b200/prototype
            ${CMAKE_SOURCE_DIR}/csrc
            ${TORCH_INCLUDE_DIRS}
            ${Python_INCLUDE_DIRS}
        )
        target_compile_definitions(fastvideo_kernel_tk_sm100_objs PRIVATE
            TORCH_EXTENSION_NAME=fastvideo_kernel_ops
            KITTENS_BLACKWELL
            TK_COMPILE_BLOCK_SPARSE_SM100
        )
        target_compile_options(fastvideo_kernel_tk_sm100_objs PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
        )
        target_sources(fastvideo_kernel_ops PRIVATE $<TARGET_OBJECTS:fastvideo_kernel_tk_sm100_objs>)
        target_compile_definitions(fastvideo_kernel_ops PRIVATE TK_COMPILE_BLOCK_SPARSE_SM100)
    endif()

    # Link against Torch libraries to avoid undefined symbols at import time
    # (e.g., torch::autograd vtables) when loading the extension module.
    target_link_libraries(fastvideo_kernel_ops PRIVATE ${TORCH_LIBRARIES})

    # Also link against libtorch_python to satisfy Python-binding symbols
    # (e.g., torch::PyWarningHandler) required by torch/extension.h.
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import torch; from pathlib import Path; p=Path(torch.__file__).parent/'lib'; m=sorted(p.glob('libtorch_python*')); print(str(m[0]) if m else '')"
        OUTPUT_VARIABLE TORCH_PYTHON_LIBRARY_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(TORCH_PYTHON_LIBRARY_PATH)
        message(STATUS "TORCH_PYTHON_LIBRARY_PATH: ${TORCH_PYTHON_LIBRARY_PATH}")
        target_link_libraries(fastvideo_kernel_ops PRIVATE "${TORCH_PYTHON_LIBRARY_PATH}")
    else()
        message(WARNING "Could not locate libtorch_python; fastvideo_kernel_ops may fail to import.")
    endif()

    # Link CUDA runtime + driver explicitly (fixes missing symbols like cuGetErrorString at import time)
    if(NOT GPU_BACKEND STREQUAL "ROCM")
        target_link_libraries(fastvideo_kernel_ops PRIVATE CUDA::cudart CUDA::cuda_driver)
    endif()

    # We install it to fastvideo_kernel/_C so we can load it to register the ops
    install(TARGETS fastvideo_kernel_ops LIBRARY DESTINATION fastvideo_kernel/_C)
endif()

