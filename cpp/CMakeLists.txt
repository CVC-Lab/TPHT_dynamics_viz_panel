# The TPHT panel's C++ backend: one simulation core (MuJoCo's C API plus the
# reference controllers) built two ways.
#   native      libtpht_core, the tpht-sim tool and the googletest suite
#   emscripten  tpht_backend.{js,wasm}, the module the page's worker loads
# Every dependency comes from cvcpkg: mujoco (native or wasm32), nlohmann-json,
# googletest, cmake, ninja and, for the browser build, emsdk. See the README.
cmake_minimum_required(VERSION 3.22)
project(tpht_backend VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TPHT_BUILD_TESTS "Build the googletest suite (native only)" ON)

find_package(mujoco 4.0 REQUIRED CONFIG)
find_package(nlohmann_json 3.11 REQUIRED CONFIG)

add_library(tpht_core STATIC
  src/backend.cpp
  src/bvls.cpp
  src/gait.cpp
  src/linalg.cpp
  src/native_controller.cpp
  src/sim.cpp)
target_include_directories(tpht_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(tpht_core PUBLIC mujoco::mujoco nlohmann_json::nlohmann_json)
if(MSVC)
  target_compile_options(tpht_core PRIVATE /W4 /permissive- /utf-8)
  target_compile_definitions(tpht_core PUBLIC _USE_MATH_DEFINES NOMINMAX)
else()
  target_compile_options(tpht_core PRIVATE -Wall -Wextra)
endif()

include(GNUInstallDirs)

if(EMSCRIPTEN)
  # The page's module: an ES module factory (export default) for a Web Worker,
  # single-threaded so GitHub Pages can serve it without COOP/COEP.
  add_executable(tpht_backend wasm/bindings.cpp)
  target_link_libraries(tpht_backend PRIVATE tpht_core)
  set_target_properties(tpht_backend PROPERTIES SUFFIX ".js")
  target_link_options(tpht_backend PRIVATE
    -lembind
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=loadTphtBackend
    -sENVIRONMENT=web,worker
    -sALLOW_MEMORY_GROWTH=1
    -sINITIAL_MEMORY=33554432
    -sSTACK_SIZE=1048576
    -sFILESYSTEM=0
    -sDISABLE_EXCEPTION_CATCHING=0)
  target_compile_options(tpht_backend PRIVATE -sDISABLE_EXCEPTION_CATCHING=0)
  target_compile_options(tpht_core PRIVATE -sDISABLE_EXCEPTION_CATCHING=0)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/build-info.json"
    "{\"version\": \"${PROJECT_VERSION}\", \"mujoco\": \"${mujoco_VERSION}\", \"emscripten\": \"${EMSCRIPTEN_VERSION}\"}\n")
  install(FILES
    $<TARGET_FILE_DIR:tpht_backend>/tpht_backend.js
    $<TARGET_FILE_DIR:tpht_backend>/tpht_backend.wasm
    "${CMAKE_CURRENT_BINARY_DIR}/build-info.json"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/tpht-backend/wasm)
else()
  add_executable(tpht-sim tools/tpht_sim.cpp)
  target_link_libraries(tpht-sim PRIVATE tpht_core)
  if(MSVC)
    target_compile_options(tpht-sim PRIVATE /W4 /permissive- /utf-8)
  endif()
  if(APPLE)
    set_target_properties(tpht-sim PROPERTIES INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
  elseif(UNIX)
    set_target_properties(tpht-sim PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
  endif()
  install(TARGETS tpht-sim RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(TARGETS tpht_core ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY include/tpht DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  if(TPHT_BUILD_TESTS)
    find_package(GTest 1.12 REQUIRED CONFIG)
    enable_testing()
    add_executable(tpht_tests tests/numerics_test.cpp tests/sim_test.cpp)
    target_link_libraries(tpht_tests PRIVATE tpht_core GTest::gtest GTest::gtest_main)
    get_filename_component(_tpht_repo "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    target_compile_definitions(tpht_tests PRIVATE TPHT_SOURCE_DIR="${_tpht_repo}")
    if(MSVC)
      target_compile_options(tpht_tests PRIVATE /utf-8)
    endif()
    include(GoogleTest)
    # PRE_TEST: listing the tests at build time would need mujoco.dll and the
    # gtest DLLs on PATH while building.
    gtest_discover_tests(tpht_tests DISCOVERY_MODE PRE_TEST)
  endif()
endif()
