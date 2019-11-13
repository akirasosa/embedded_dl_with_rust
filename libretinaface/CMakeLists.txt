set(COMMON_SOURCES
        src/logger.cpp
        src/engine.cpp
        )

find_package(CUDA 10.0 REQUIRED)
find_package(TensorRT REQUIRED)

add_library(retinaface STATIC src/retinaface.cpp ${COMMON_SOURCES})

target_include_directories(
        retinaface
        PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(
        retinaface
        ${CUDA_LIBRARIES}
        ${TENSORRT_LIBRARY}
)

install(TARGETS retinaface DESTINATION .)
