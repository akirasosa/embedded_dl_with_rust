extern crate itertools;
extern crate itertools_num;

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::time::SystemTime;

use itertools_num::linspace;
use opencv::core::{Mat, Point, Rect, Scalar, ToInputOutputArray};
use opencv::imgcodecs::{imread, IMREAD_COLOR, imwrite};
use opencv::imgproc::LINE_8;
use opencv::prelude::Vector;
use opencv::Result;
use opencv::types::VectorOfint;
use opencv::videoio::{CAP_ANY, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, VideoCapture, VideoWriter};

use embedded_dl_with_rust::retinaface::{Detection, DetectionResult, RetinaFace};
use clap::{App, Arg};

const MODEL_PATH: &str = "tmp/retinaface_resnet50_trained_opt.trt";

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
    where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn main() {
//    if let Ok(lines) = read_lines("./tmp/list.txt") {
//        for line in lines {
//            if let Ok(ip) = line {
//                println!("{}", ip);
//            }
//        }
//    }
    let matches = App::new("My Super Program")
        .version("1.0")
        .author("Kevin K. <kbknapp@gmail.com>")
        .about("Does awesome things")
        .arg(Arg::with_name("list_file")
            .long("list_file")
            .required(true)
            .takes_value(true))
        .arg(Arg::with_name("T")
            .long("T")
            .default_value("4")
            .takes_value(true))
        .get_matches();
    println!("{:?}", matches.value_of("list_file"));
    println!("{:?}", matches.value_of("T"));
//    test_video().unwrap();

//    env_logger::init();
//    run().unwrap();
//    run_video().unwrap();
}

#[allow(unused)]
fn run() -> Result<()> {
    let mut img = imread("images/faces.jpg", IMREAD_COLOR)?;

    let mut retinaface = RetinaFace::new(MODEL_PATH);
    let detections = unsafe { retinaface.detect(&img) };

    render(&mut img, detections);
    imwrite("tmp/out.jpg", &img, &VectorOfint::new())?;

    Ok(())
}

#[allow(unused)]
fn test_video() -> Result<()> {
    let video_in_path = "/mnt/lvstuff/akirasosa/data/deepfake-detection-challenge/test_videos/scbdenmaed.mp4";
    let mut cap = VideoCapture::new_from_file_with_backend(video_in_path, CAP_ANY)?;

    let codec = VideoWriter::fourcc('M' as i8, 'J' as i8, 'P' as i8, 'G' as i8)?;
    let fps = cap.get(CAP_PROP_FPS)?;
    let frame_count = cap.get(CAP_PROP_FRAME_COUNT)?;
    let frames_to_use = {
        let double = linspace::<f64>(0., frame_count, 4 + 2)
            .map(|n| n as i64)
            .map(|n| [n, n + 1])
            .collect::<Vec<_>>();
        double[1..double.len() - 1].iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>()
    };

    let mut retinaface = RetinaFace::new(MODEL_PATH);

    let start = SystemTime::now();
    for n in 0..frame_count as i64 {
        if &n > frames_to_use.last().unwrap() {
            break;
        }

        if !frames_to_use.contains(&n) {
            if let Ok(result) = cap.grab() {
                continue;
            } else {
                break;
            }
        }

        let mut frame = Mat::default().unwrap();
        if let Err(result) = cap.read(&mut frame) {
            break;
        }

        let detections = unsafe { retinaface.detect(&frame) };
        render(&mut frame, detections);
        imwrite(&format!("out/{:03}.jpg", n), &frame, &VectorOfint::new())?;

        println!("{}", n);
    }
    let end = SystemTime::now();
    let elapsed = end.duration_since(start).unwrap();
    println!("elapsed {:?}", elapsed);

    cap.release()?;

    Ok(())
}

#[allow(unused)]
fn run_video() -> Result<()> {
    let video_in_path = "/home/akirasosa/tmp/face-demographics-walking.mp4";
    let mut cap = VideoCapture::new_from_file_with_backend(video_in_path, CAP_ANY)?;

    let video_out_path = "tmp/out.avi";
    let codec = VideoWriter::fourcc('M' as i8, 'J' as i8, 'P' as i8, 'G' as i8)?;
    let fps = cap.get(CAP_PROP_FPS)?;
    let size = Mat::default().and_then(|mut mat| {
        cap.read(&mut mat);
        mat.size()
    })?;
    let mut writer = VideoWriter::new(video_out_path, codec, fps, size, true)?;

    let mut retinaface = RetinaFace::new(MODEL_PATH);

    loop {
        let mut frame = Mat::default().and_then(|mut mat| {
            cap.read(&mut mat);
            Ok(mat)
        })?;

        if frame.empty()? { break; }

        let detections = unsafe { retinaface.detect(&frame) };
        render(&mut frame, detections);

        writer.write(&frame).unwrap();
    }

    cap.release()?;
    writer.release()?;

    Ok(())
}

fn color(b: u8, g: u8, r: u8) -> Scalar {
    Scalar::new(b as f64, g as f64, r as f64, 0.)
}

fn circle(img: &mut dyn ToInputOutputArray, c: Point, radius: i32, color: Scalar) {
    opencv::imgproc::circle(img, c, radius, color, 2, LINE_8, 0).unwrap()
}

fn rectangle(img: &mut dyn ToInputOutputArray, rec: Rect, color: Scalar) {
    opencv::imgproc::rectangle(img, rec, color, 2, LINE_8, 0).unwrap();
}

fn render(mut img: &mut Mat, detections: DetectionResult) {
    detections.into_iter().for_each(|d: Detection| {
        rectangle(&mut img, d.bbox, color(0, 0, 255));
        circle(&mut img, d.p0, 1, color(0, 0, 255));
        circle(&mut img, d.p1, 1, color(0, 255, 0));
        circle(&mut img, d.p2, 1, color(255, 0, 0));
        circle(&mut img, d.p3, 1, color(0, 255, 255));
        circle(&mut img, d.p4, 1, color(255, 255, 0));
    });
}
