cmake_minimum_required(VERSION 3.18)
project("macademy_cpp" VERSION 1.0.0)

# Config

#Backends
set(MACADEMY_BACKEND_OPENCL 1)

#-----------------------------

add_library(${PROJECT_NAME}
    include/common.h
    include/i_compute_backend.h
    include/cpu_compute_backend.h
    include/network.h
    include/i_weight_initializer.h
    include/default_weight_initializer.h
    
    src/cpu_compute_backend.cpp
    src/network.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}/include 
    )

# target_link_libraries(${PROJECT_NAME} PUBLIC )

add_library(::${PROJECT_NAME} ALIAS ${PROJECT_NAME}) 