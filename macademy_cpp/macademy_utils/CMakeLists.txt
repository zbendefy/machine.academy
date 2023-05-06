cmake_minimum_required(VERSION 3.18)
project("macademy_utils" VERSION 1.0.0)

# Config

add_library(${PROJECT_NAME}
    include/macademy_utils/console_app.h

    src/console_app.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC 
    ${CMAKE_CURRENT_LIST_DIR}/include 
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ::macademy_cpp
)

add_library(::${PROJECT_NAME} ALIAS ${PROJECT_NAME}) 