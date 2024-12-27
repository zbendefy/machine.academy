cmake_minimum_required(VERSION 3.18)
project("macademy_cpp" VERSION 1.0.0)

# Config

#Backends
set(MACADEMY_BACKEND_OPENCL 1)
set(MACADEMY_BACKEND_VULKAN 1)

#-----------------------------

if (MACADEMY_BACKEND_OPENCL)
    message("OPENCL_SDK env var is set to: '$ENV{OPENCL_SDK}'")
    
    file(GLOB_RECURSE OPENCL_SOURCE_FILES 
        include/opencl_backend/*.h
        src/opencl_backend/*.cpp
    )
    set(OPENCL_INCLUDE_DIRS $ENV{OPENCL_SDK}/include)
    set(OPENCL_LINK_DIRECTORY $ENV{OPENCL_SDK}/lib)
    set(OPENCL_LINK_LIBS OpenCL)
else()
    set(OPENCL_SOURCE_FILES )
    set(OPENCL_LINK_DIRECTORY )
    set(OPENCL_LINK_LIBS )
    set(OPENCL_INCLUDE_DIRS )
endif()

if (MACADEMY_BACKEND_VULKAN)
    message("VULKAN_SDK env var is set to: '$ENV{VULKAN_SDK}'")
    message("RENDERDOC_PATH env var is set to: '$ENV{RENDERDOC_PATH}'")

    file(GLOB_RECURSE VULKAN_SOURCE_FILES 
        include/vulkan_backend/*.h
        include/vulkan_backend/shaders/*.h

        src/vulkan_backend/*.cpp
    )
    set(VULKAN_INCLUDE_DIRS 
            $ENV{VULKAN_SDK}/include
            ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/VulkanMemoryAllocator/src
            $ENV{RENDERDOC_PATH}
    )
    set(VULKAN_LINK_DIRECTORY $ENV{VULKAN_SDK}/lib)
    set(VULKAN_LINK_LIBS ::vulkan_sdk ::spirv_reflect VulkanMemoryAllocator)
else()
    set(VULKAN_SOURCE_FILES )
    set(VULKAN_LINK_DIRECTORY )
    set(VULKAN_LINK_LIBS )
    set(VULKAN_INCLUDE_DIRS )
endif()

file(GLOB_RECURSE MACADEMY_SOURCE_FILES
    include/*.h
    include/cpu_backend/*.h

    src/*.cpp
    src/cpu_backend/*.cpp
)

add_library(${PROJECT_NAME}
    ${MACADEMY_SOURCE_FILES}
    ${OPENCL_SOURCE_FILES}
    ${VULKAN_SOURCE_FILES}
)

if (MACADEMY_BACKEND_OPENCL)
    target_compile_definitions(${PROJECT_NAME} PUBLIC MACADEMY_OPENCL_BACKEND=1)
endif()

if (MACADEMY_BACKEND_VULKAN)
    target_compile_definitions(${PROJECT_NAME} PUBLIC MACADEMY_VULKAN_BACKEND=1)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}/include 
    ${OPENCL_INCLUDE_DIRS}
    ${VULKAN_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/hwinfo/include
    )

target_link_directories(${PROJECT_NAME} PUBLIC 
    ${OPENCL_LINK_DIRECTORY}
    ${VULKAN_LINK_DIRECTORY}
)

if(UNIX)
    set(ADDITIONAL_LINK_TARGETS pthread) 
else()
    set(ADDITIONAL_LINK_TARGETS) 
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC 
    ${OPENCL_LINK_LIBS}
    ${VULKAN_LINK_LIBS}
    hwinfo::HWinfo
    nlohmann_json
    ${ADDITIONAL_LINK_TARGETS}
)

add_library(::${PROJECT_NAME} ALIAS ${PROJECT_NAME}) 