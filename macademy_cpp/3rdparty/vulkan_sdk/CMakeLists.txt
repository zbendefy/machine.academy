cmake_minimum_required(VERSION 3.18)
project("vulkan_sdk" NONE)

add_library(${PROJECT_NAME} INTERFACE)
add_library(::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

string(REPLACE "\\" "/" VulkanSDK_PATH $ENV{VULKAN_SDK})

message("Vulkan SDK found: '${VulkanSDK_PATH}'")

if(WIN32)
    target_include_directories(${PROJECT_NAME} INTERFACE ${VulkanSDK_PATH}/Include/vulkan)
    set(VULKAN_LIBS "${VulkanSDK_PATH}/Lib/vulkan-1.lib")
else()

    target_include_directories(${PROJECT_NAME} INTERFACE ${VulkanSDK_PATH}/include/vulkan)
    set(VULKAN_LIBS "${VulkanSDK_PATH}/lib/libvulkan.so")

    #required for VMA under linux
    find_package(Threads REQUIRED)
    set(THREAD_LIB "Threads::Threads")
endif()

target_link_libraries(${PROJECT_NAME} INTERFACE ${VULKAN_LIBS} ${THREAD_LIB})
