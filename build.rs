extern crate cmake;
//extern crate cpp_build;
extern crate bindgen;

use std::env;

fn main()
{
    let tensorrt_root = env::var("TENSORRT_ROOT")
        .expect("Can not find TENSORRT_ROOT env var.");
    let cuda_root = match env::var("CUDA_HOME") {
        Ok(v) => v,
        Err(_) => String::from("/usr/local/cuda"),
    };

    let cpp_libs = cmake::Config::new(".")
        .define("TENSORRT_ROOT", &tensorrt_root)
        .build();

    println!("cargo:rustc-link-search=native={}", cpp_libs.display());
    println!("cargo:rustc-link-search=native={}/lib", &tensorrt_root);
    println!("cargo:rustc-link-search=native={}/lib64", &cuda_root);
    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=static=retinaface");
    println!("cargo:rerun-if-changed=libretinaface/**/*");
    println!("cargo:rerun-if-changed=wrapper.hpp");

    let target = env::var("TARGET").unwrap();
    if target.contains("apple")
    {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if target.contains("linux")
    {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else {
        unimplemented!();
    }

    let bindings = bindgen::Builder::default()
        .header("wrapper.hpp")
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg(format!("-I{}/include", &tensorrt_root))
        .clang_arg(format!("-I{}/include", &cuda_root))
        .whitelist_type("mal::.*")
        .whitelist_function("mal::.*")
        .generate()
        .expect("Unable to generate bindings");
    bindings
        .write_to_file("src/bindings.rs")
        .expect("Couldn't write bindings!");

//    cpp_build::Config::new()
//        .include(format!("{}/include", &tensorrt_root))
//        .include(format!("{}/include", &cuda_root))
//        .include("/usr/local/include/opencv4")
//        .include("libretinaface/src")
//        .build("src/lib.rs");
}