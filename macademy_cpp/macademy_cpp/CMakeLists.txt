cmake_minimum_required(VERSION 3.18)
project("macademy_cpp" VERSION 1.0.0)

# Config

#Backends
set(MACADEMY_BACKEND_OPENCL 1)
set(MACADEMY_BACKEND_VULKAN 1)

#-----------------------------

if (MACADEMY_BACKEND_OPENCL)
    message("OPENCL_SDK env var is set to: '$ENV{OPENCL_SDK}'")
    find_package(OpenCL)
    message("OpenCL Found: " ${OpenCL_FOUND})
    
    file(GLOB_RECURSE OPENCL_SOURCE_FILES 
        include/opencl_backend/*.h
        src/opencl_backend/*.cpp
    )
    set(OPENCL_KERNELS 
        src/opencl_backend/opencl_kernels.cl
    )
    if (NOT "$ENV{OPENCL_SDK}" STREQUAL "")
        message("  Using OpenCL SDK dir: " $ENV{OPENCL_SDK})
        set(OPENCL_INCLUDE_DIRS $ENV{OPENCL_SDK}/include)
        set(OPENCL_LINK_DIRECTORY $ENV{OPENCL_SDK}/lib)
        set(OPENCL_LINK_LIBS OpenCL)
    elseif(OpenCL_FOUND)
        message("  Using found OpenCL install dir: " ${OpenCL_INCLUDE_DIRS})
        set(OPENCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS})
        set(OPENCL_LINK_LIBS ${OpenCL_LIBRARIES})
        set(OPENCL_LINK_DIRECTORY)
    else()
        message("Error: OpenCL not found!!! Install opencl dev packages, or download the opencl sdk and point to it using the OPENCL_SDK env variable!")
    endif()
else()
    set(OPENCL_SOURCE_FILES )
    set(OPENCL_LINK_DIRECTORY )
    set(OPENCL_LINK_LIBS )
    set(OPENCL_INCLUDE_DIRS )
endif()

if (MACADEMY_BACKEND_VULKAN)
    message("VULKAN_SDK env var is set to: '$ENV{VULKAN_SDK}'")
    find_package(Vulkan REQUIRED)
    message("  Vulkan found: " ${Vulkan_FOUND} " GLSLC found: " ${Vulkan_glslc_FOUND} "  Include dirs: " ${Vulkan_INCLUDE_DIRS})
    message("RENDERDOC_PATH env var is set to: '$ENV{RENDERDOC_PATH}'")
    
    if(DEFINED ENV{RENDERDOC_PATH})
        add_compile_definitions(RDOC_ENABLED=1)
        message(" (RenderDoc debugging: enabled)")
    else()
        add_compile_definitions(RDOC_ENABLED=0)
        message(" (RenderDoc debugging: disabled)")
    endif()

    file(GLOB_RECURSE VULKAN_SOURCE_FILES 
        include/vulkan_backend/*.h
        include/vulkan_backend/shaders/*.h

        src/vulkan_backend/*.cpp
    )
    set(VULKAN_SHADERS 
        include/vulkan_backend/shaders/kernel_calc_single_layer.glsl
        include/vulkan_backend/shaders/kernel_training_forward_pass.glsl
        include/vulkan_backend/shaders/kernel_training_backward_pass.glsl
        include/vulkan_backend/shaders/kernel_training_backward_pass_swadd.glsl
        include/vulkan_backend/shaders/kernel_apply_gradient.glsl
    )
    set(VULKAN_INCLUDE_DIRS 
            ${Vulkan_INCLUDE_DIRS}
            ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/VulkanMemoryAllocator/src
            $ENV{RENDERDOC_PATH}
    )
    set(VULKAN_LINK_LIBS ${Vulkan_LIBRARIES} VulkanMemoryAllocator)
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

if(MACADEMY_BACKEND_OPENCL)
    find_package(Python COMPONENTS Interpreter REQUIRED)
    add_custom_target(
    compile_cl_kernels ALL
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/opencl_backend/compile_kernel.py ${CMAKE_CURRENT_SOURCE_DIR} ${OPENCL_KERNELS}
    BYPRODUCTS ${OPENCL_KERNELS}
    COMMENT "Compiling opencl kernels"
    )

    add_dependencies(${PROJECT_NAME} compile_cl_kernels )
endif()

if(MACADEMY_BACKEND_VULKAN)
    find_package(Python COMPONENTS Interpreter REQUIRED)
    add_custom_target(
    compile_vk_shaders ALL
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/include/vulkan_backend/shaders/compile_shaders.py ${CMAKE_CURRENT_SOURCE_DIR} ${Vulkan_GLSLC_EXECUTABLE} ${VULKAN_SHADERS}
    BYPRODUCTS ${VULKAN_SHADERS}
    COMMENT "Compiling vulkan shaders"
    )

    add_dependencies(${PROJECT_NAME} compile_vk_shaders)
endif()