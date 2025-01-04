cmake_minimum_required(VERSION 3.18)
project("poly_inverse" VERSION 1.0.0)

add_executable(${PROJECT_NAME}
    main.cpp
)


target_link_libraries(${PROJECT_NAME} PUBLIC
    ::macademy_cpp
    ::macademy_utils
)