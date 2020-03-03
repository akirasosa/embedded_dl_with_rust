cmake_minimum_required(VERSION 3.10)
project(RustTensorRTExample
        VERSION 0.1
        LANGUAGES CXX
        )

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif ()

add_subdirectory(libretinaface)


