# Rust for Embedded Deep Learning Model

This project is a starter code for embedded deep learning model with [Rust lang](https://www.rust-lang.org/).

## Motivation

Though cloud usage is common in most situations, it sometimes necessary to run deep learning model in restricted environment such as embedded systems.

C++ is the most widely used for embedded system, but its productivity is low and it sometimes ships unsafe code. Rust can give developers both of performance and safety.

This repository uses C++ only for minimal usage which interacts with GPU using TensorRT. The rest of codes are all written by Rust. It's useful to make some apps which use GPU in embedded system.

## Requirements

* Rust
  * [rust-cpp](https://github.com/mystor/rust-cpp)
  * [cmake-rs](https://github.com/alexcrichton/cmake-rs)
* [TensorRT 6](https://developer.nvidia.com/tensorrt)
* OpenCV4


## Getting Started

Download TensorRT 6 from [Nvidia page](https://developer.nvidia.com/tensorrt) and install it by following the [guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing-tar) as below.

```
$ tar xzvf TensorRT-6.x.x.x.<os>.<arch>-gnu.cuda-x.x.cudnn7.x.tar.gz
$ ls TensorRT-6.x.x.x
bin  data  doc  graphsurgeon  include  lib  python  samples  targets  TensorRT-Release-Notes.pdf  uff
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<eg:TensorRT-6.x.x.x/lib>
```

We use headers and lib in it later. So, let's set ```TENSORRT_ROOT``` env var.

```
$ export TENSORRT_ROOT=/PATH/to/TensorRT-6.x.x.x
```

Install OpenCV4. It's supposed to be installed under ```/usr/local```. You will see something like below.

```
$ ls /usr/local/include/opencv4
opencv2
$ opencv_version
4.1.0
```

Install Rust by following the [guide](https://www.rust-lang.org/tools/install) and confirm it.

```
$ cargo --version
cargo 1.38.0 (23ef9a4ef 2019-08-20)
```

Download pre-trained TensorRT serialized model [here](https://www.dropbox.com/s/1uen1ow8e4vap7m/retinaface_resnet50_trained_opt.trt?dl=0) and put it in ```tmp``` dir under project root.
```
$ ls tmp/
retinaface_resnet50_trained_opt.trt
```
This model uses Resnet50 as a backbone, which is not so fast.

It's ready to run.
```
$ cargo run
[10/07/2019-17:28:03] [I] [TRT] Init Engine from tmp/retinaface_resnet50_trained_opt.trt
[10/07/2019-17:28:06] [W] [TRT] TensorRT was linked against cuDNN 7.6.3 but loaded cuDNN 7.6.0
[10/07/2019-17:28:06] [W] [TRT] TensorRT was linked against cuDNN 7.6.3 but loaded cuDNN 7.6.0
[10/07/2019-17:28:06] [I] [TRT] Engine Destroyed
$ ls tmp/out.jpg
tmp/out.jpg
```

The output will be something like this.

![result](./images/result.jpg)
