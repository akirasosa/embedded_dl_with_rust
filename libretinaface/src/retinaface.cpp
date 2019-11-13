#include "retinaface.hpp"

constexpr size_t kNDet = 16128;

mal::RetinaFace::RetinaFace(const char *modelPath) : engine_(modelPath) {
}

mal::Detection mal::RetinaFace::detect(const float *data, const size_t size) {
    // Create device buffers
    const auto inputSize = size * sizeof(float);
    void *imgBuf, *scoresBuf, *bboxesBuf, *landmarksBuf;
    cudaMalloc(&imgBuf, inputSize * sizeof(float));
    cudaMalloc(&scoresBuf, kNDet * sizeof(float));
    cudaMalloc(&bboxesBuf, kNDet * 4 * sizeof(float));
    cudaMalloc(&landmarksBuf, kNDet * 10 * sizeof(float));

    // Copy image to device
    cudaMemcpy(imgBuf, data, size * sizeof(float), cudaMemcpyHostToDevice);

    // Infer
    std::vector<void *> buffers = {imgBuf, bboxesBuf, landmarksBuf, scoresBuf};  // This order can not be changed.
    engine_.infer(buffers);

    // Get back the outputs
    float scoresOut[kNDet];
    float bboxesOut[kNDet * 4];
    float landmarksOut[kNDet * 10];
    cudaMemcpy(scoresOut, scoresBuf, sizeof(float) * kNDet, cudaMemcpyDeviceToHost);
    cudaMemcpy(bboxesOut, bboxesBuf, sizeof(float) * kNDet * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(landmarksOut, landmarksBuf, sizeof(float) * kNDet * 10, cudaMemcpyDeviceToHost);

    return Detection{scoresOut, bboxesOut, landmarksOut, kNDet};
}


mal::RetinaFace::~RetinaFace() = default;
