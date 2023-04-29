cmake_minimum_required(VERSION 3.18)

project("macademy_cpp_sln" VERSION 0.1.0)

message("CMake version: " ${CMAKE_SYSTEM_NAME})
message("Compiler: " ${CMAKE_CXX_COMPILER_ID})
message("System: " ${CMAKE_SYSTEM})
message("System processor: " ${CMAKE_SYSTEM_PROCESSOR})
message("Host system processor: " ${CMAKE_HOST_SYSTEM_PROCESSOR})

set(COMPILE_TESTS false CACHE BOOL "Compile tests")
set(COMPILE_TESTS true)

include(compiler_flags.cmake)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/3rdparty)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/macademy_cpp)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/console)

if(COMPILE_TESTS)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif()