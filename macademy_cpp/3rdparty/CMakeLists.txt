cmake_minimum_required(VERSION 3.18)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/json)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/hwinfo)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/vulkan_sdk)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/VulkanMemoryAllocator)

if(COMPILE_TESTS)
    set(BUILD_GMOCK false)
    set(INSTALL_GTEST false)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/googletest)
endif()

