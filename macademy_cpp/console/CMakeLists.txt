cmake_minimum_required(VERSION 3.18)
project("console" VERSION 1.0.0)

add_executable(${PROJECT_NAME}
    console_app.h

    main.cpp
)


target_link_libraries(${PROJECT_NAME} PUBLIC
    ::macademy_cpp
)