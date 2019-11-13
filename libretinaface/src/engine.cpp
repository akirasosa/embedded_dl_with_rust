#include <NvOnnxParser.h>
#include <iostream>
#include <fstream>
#include "engine.hpp"
#include "logger.h"


mal::Engine::Engine(const string &path) {
    auto &logger = gLogger.getTRTLogger();
    const auto msg = "Init Engine from " + path;
    logger.log(Severity::kINFO, msg.c_str());

//    _runtime = createInferRuntime(logger);
//
//    auto builder = createInferBuilder(logger);
//    builder->setMaxBatchSize(1);
//
//    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
//    const auto network = builder->createNetworkV2(explicitBatch);
//
//    const auto config = builder->createBuilderConfig();
//    config->setMaxWorkspaceSize(16 * 1024 * 1024);
//    config->setFlag(BuilderFlag::kFP16);
//
//    constexpr auto MODEL_PATH = "/home/akirasosa/tmp/retinaface_resnet50_trained_opt.onnx";
//    const auto parser = nvonnxparser::createParser(*network, logger);
//    parser->parseFromFile(MODEL_PATH, static_cast<int>(gLogger.getReportableSeverity()));
//
//    _engine = builder->buildEngineWithConfig(*network, *config);
//    _context = _engine->createExecutionContext();
//    cudaStreamCreate(&_stream);

    _runtime = createInferRuntime(logger);
    _load(path);
    _prepare();
}

mal::Engine::~Engine() {
    if (_stream) cudaStreamDestroy(_stream);
    if (_context) _context->destroy();
    if (_engine) _engine->destroy();
    if (_runtime) _runtime->destroy();

    auto &logger = gLogger.getTRTLogger();

    logger.log(Severity::kINFO, "Engine Destroyed");
}

void mal::Engine::infer(vector<void *> &buffers) {
    _context->enqueueV2(buffers.data(), _stream, nullptr);
    cudaStreamSynchronize(_stream);
}


void mal::Engine::_load(const string &path) {
    ifstream file(path, ios::in | ios::binary);
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    char *buffer = new char[size];
    file.read(buffer, size);
    file.close();

    _engine = _runtime->deserializeCudaEngine(buffer, size, nullptr);

    delete[] buffer;
}

void mal::Engine::_prepare() {
    _context = _engine->createExecutionContext();
    cudaStreamCreate(&_stream);
}

