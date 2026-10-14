# Strata on Apple silicon - the Metal build of the engine (EXPERIMENTAL, docs/APPLE_METAL.md).
#
# A separate project, like sycl/: the top-level build describes the CUDA / HIP engine, whose kernels are CUDA C++.
# This one takes the model graph and the Metal kernels from llama.cpp (ggml-metal, the qwen4exp architecture) at a
# pinned commit, with the patches in patches/ applied to a copy inside the build folder, and builds Strata's engine
# program (src/main.cpp) on top: the memory plan for unified memory, the expert cache's start-up fill, the read-only
# SSD tier and the `--serve` protocol of serve/server.py.
#
#   cmake -S metal -B build-metal -DCMAKE_BUILD_TYPE=Release && cmake --build build-metal -j
#   (the same as -DSTRATA_ENABLE_METAL=ON on the top-level project)
#
# Needs: macOS on Apple silicon, the Xcode command line tools (clang, the Metal framework), CMake 3.24+, git.
# Xcode itself is not needed: the Metal shaders are compiled from source when the engine starts.
cmake_minimum_required(VERSION 3.24)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt _strata_proj REGEX "^project\\(strata VERSION [0-9.]+")
string(REGEX MATCH "VERSION ([0-9.]+)" _m "${_strata_proj}")
if(CMAKE_MATCH_1)
  set(STRATA_METAL_VERSION ${CMAKE_MATCH_1})
else()
  set(STRATA_METAL_VERSION 0.1.42)
endif()
project(strata_metal VERSION ${STRATA_METAL_VERSION} LANGUAGES C CXX)
if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  message(FATAL_ERROR "the Metal engine is for Apple silicon Macs (this is ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR})")
endif()
set(STRATA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- llama.cpp: the pinned commit, patched in the build folder (the checkout it comes from is never edited)
set(STRATA_METAL_LLAMA_COMMIT 23b0202a189c44a54625aadcb37a946dd1d6278d)   # 2026-10-10: qwen4exp, the MoE cache, lazy PLE rows
set(STRATA_LLAMA_DIR "" CACHE PATH "a llama.cpp git checkout that has the pinned commit (empty: ../llama.cpp beside Strata if it is there, else GitHub)")
if(NOT STRATA_LLAMA_DIR AND EXISTS ${STRATA_ROOT}/../llama.cpp/.git)
  get_filename_component(STRATA_LLAMA_DIR ${STRATA_ROOT}/../llama.cpp ABSOLUTE)
endif()
find_package(Git REQUIRED)
set(_llama ${CMAKE_BINARY_DIR}/llama.cpp)
file(GLOB _patches ${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch)
list(SORT _patches)
set(_stamp_want "${STRATA_METAL_LLAMA_COMMIT}")
foreach(p ${_patches})
  file(SHA256 ${p} _h)
  string(APPEND _stamp_want " ${_h}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${p})
endforeach()
set(_stamp_have "")
if(EXISTS ${_llama}/.strata-stamp)
  file(READ ${_llama}/.strata-stamp _stamp_have)
endif()
function(_strata_git)
  execute_process(COMMAND ${GIT_EXECUTABLE} ${ARGN} WORKING_DIRECTORY ${_llama} RESULT_VARIABLE _r OUTPUT_QUIET ERROR_VARIABLE _e)
  if(NOT _r EQUAL 0)
    message(FATAL_ERROR "llama.cpp for the Metal engine: git ${ARGN} failed:\n${_e}")
  endif()
endfunction()
if(NOT _stamp_have STREQUAL _stamp_want)
  if(NOT EXISTS ${_llama}/.git)
    file(MAKE_DIRECTORY ${_llama})
    _strata_git(init -q)
    if(STRATA_LLAMA_DIR)
      message(STATUS "llama.cpp ${STRATA_METAL_LLAMA_COMMIT} from ${STRATA_LLAMA_DIR}")
      _strata_git(fetch -q ${STRATA_LLAMA_DIR} ${STRATA_METAL_LLAMA_COMMIT})
    else()
      message(STATUS "llama.cpp ${STRATA_METAL_LLAMA_COMMIT} from GitHub")
      _strata_git(fetch -q --depth 1 https://github.com/ggml-org/llama.cpp ${STRATA_METAL_LLAMA_COMMIT})
    endif()
  endif()
  # back to the pinned tree, then the patches: only the files they touch change, so only those are compiled again
  _strata_git(checkout -q -f ${STRATA_METAL_LLAMA_COMMIT})
  foreach(p ${_patches})
    message(STATUS "llama.cpp patch: ${p}")
    _strata_git(apply --whitespace=nowarn ${p})
  endforeach()
  file(WRITE ${_llama}/.strata-stamp "${_stamp_want}")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)   # one self-contained `strata`
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
# strata-vision (tools/vision): the image encoder, llama.cpp's mtmd library on the same ggml as the engine
option(STRATA_METAL_VISION "Build strata-vision, the image encoder" ON)
if(STRATA_METAL_VISION)
  set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
  set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
endif()
add_subdirectory(${_llama} ${CMAKE_BINARY_DIR}/llama EXCLUDE_FROM_ALL)

add_executable(strata src/main.cpp src/gpu_keepalive.mm)
# Objective-C++ through the C++ compiler: enabling the OBJCXX language for the project changes how ggml's .m files
# are compiled
set_source_files_properties(src/gpu_keepalive.mm PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x objective-c++ -fobjc-arc")
target_compile_definitions(strata PRIVATE STRATA_VERSION="${PROJECT_VERSION}-metal")
target_compile_options(strata PRIVATE -Wall -Wextra)
target_link_libraries(strata PRIVATE llama ggml "-framework Metal" "-framework Foundation")
set_target_properties(strata PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(STRATA_METAL_VISION)
  add_executable(strata-vision ${STRATA_ROOT}/tools/vision/strata_vision.cpp)
  target_link_libraries(strata-vision PRIVATE mtmd llama)
  target_compile_features(strata-vision PRIVATE cxx_std_17)
  set_target_properties(strata-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

option(STRATA_METAL_BUILD_TESTS "Build Metal kernel regression tests" OFF)
if(STRATA_METAL_BUILD_TESTS)
  enable_testing()
  add_executable(strata-test-indexed-qsa tests/indexed_qsa_test.cpp)
  target_link_libraries(strata-test-indexed-qsa PRIVATE ggml)
  add_test(NAME metal-indexed-qsa COMMAND strata-test-indexed-qsa)

  add_executable(strata-test-backend-ops ${_llama}/tests/test-backend-ops.cpp)
  target_include_directories(strata-test-backend-ops PRIVATE ${_llama}/ggml/src)
  target_link_libraries(strata-test-backend-ops PRIVATE ggml)
  add_test(NAME metal-top-k COMMAND strata-test-backend-ops -b MTL0 -o TOP_K)
endif()

# serve/server.py reads the engine's version from the BUILD.json beside it
file(WRITE ${CMAKE_BINARY_DIR}/bin/BUILD.json
     "{\"version\": \"${PROJECT_VERSION}\", \"backend\": \"metal\", \"llama_cpp\": \"${STRATA_METAL_LLAMA_COMMIT}\"}\n")
