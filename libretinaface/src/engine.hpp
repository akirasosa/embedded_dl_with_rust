#pragma once

#include <string>
#include <vector>

#include <NvInfer.h>

#include <cuda_runtime.h>
#include <memory>

using namespace std;
using namespace nvinfer1;

namespace mal {

    class Engine {
    public:
        explicit Engine(const std::string &engine_path);

        ~Engine();

        void infer(vector<void *> &buffers);

    private:
        ICudaEngine *_engine = nullptr;
        IRuntime *_runtime = nullptr;
        IExecutionContext *_context = nullptr;
        cudaStream_t _stream = nullptr;

        void _load(const string &path);

        void _prepare();

    };

}