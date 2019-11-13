#pragma once

#include "engine.hpp"

namespace mal {
    struct Detection {
        float *const scores;
        float *const bboxes;
        float *const landmarks;
        int32_t const nDet;
    };

    class RetinaFace {
    public:
        explicit RetinaFace(const char *modelPath);

        ~RetinaFace();

        Detection detect(const float *data, size_t size);

    private:
        mal::Engine engine_;
    };

}
