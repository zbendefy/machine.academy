cmake_minimum_required(VERSION 3.18)
project("macademy_cpp" VERSION 1.0.0)

# Config

#Backends
set(MACADEMY_BACKEND_OPENCL 1)

#-----------------------------

if (MACADEMY_BACKEND_OPENCL)
    add_compile_definitions(MACADEMY_OPENCL_BACKEND=1)
    message("OPENCL_SDK env var is set to: '$ENV{OPENCL_SDK}'")

    set(OPENCL_SOURCE_FILES 
        include/opencl_backend/opencl_compute_device.h
        include/opencl_backend/opencl_common.h
        include/opencl_backend/opencl_buffer.h

        src/opencl_backend/opencl_compute_device.cpp
    )
    set(OPENCL_INCLUDE_DIRS $ENV{OPENCL_SDK}/include)
    set(OPENCL_LINK_DIRECTORY $ENV{OPENCL_SDK}/lib)
    set(OPENCL_LINK_LIBS opencl)
else()
    set(OPENCL_SOURCE_FILES )
    set(OPENCL_LINK_DIRECTORY )
    set(OPENCL_LINK_LIBS )
    set(OPENCL_INCLUDE_DIRS )
endif()

add_library(${PROJECT_NAME}
    include/common.h
    include/training.h
    include/training_suite.h
    include/i_compute_backend.h
    include/cpu_compute_backend.h
    include/network.h
    include/i_weight_initializer.h
    include/default_weight_initializer.h
    include/utils.h
    
    src/cpu_compute_backend.cpp
    src/network.cpp
    src/training.cpp
    src/utils.cpp

    ${OPENCL_SOURCE_FILES}
)

target_include_directories(${PROJECT_NAME} PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}/include 
    ${OPENCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/hwinfo/include
    )

target_link_directories(${PROJECT_NAME} PUBLIC 
    ${OPENCL_LINK_DIRECTORY}
)

target_link_libraries(${PROJECT_NAME} PUBLIC 
    ${OPENCL_LINK_LIBS}
    hwinfo::HWinfo
    nlohmann_json
)

add_library(::${PROJECT_NAME} ALIAS ${PROJECT_NAME}) 