cmake_minimum_required(VERSION 3.18)

enable_testing()

project("macademy_cpp_test")

# For Windows: Prevent overriding the parent project's compiler/linker settings

add_executable(${PROJECT_NAME}
    test_main.cpp
    test_training.cpp
    test_compute_devices.cpp
)
    
target_include_directories(${PROJECT_NAME} PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../macademy_cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/googletest/googletest/include
    )
    
target_link_libraries(${PROJECT_NAME} PUBLIC
    ::macademy_cpp
    gtest_main
)
    
include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})