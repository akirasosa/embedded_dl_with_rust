extern crate itertools;
extern crate itertools_num;

use std::fs;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::time::SystemTime;

use clap::{App, Arg};
use failure::Error;
use itertools_num::linspace;
use opencv::core::{Mat, Point, Rect, Scalar, ToInputOutputArray};
use opencv::imgcodecs::{imread, IMREAD_COLOR, imwrite};
use opencv::imgproc::LINE_8;
use opencv::prelude::Vector;
use opencv::types::VectorOfint;
use opencv::videoio::{CAP_ANY, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, VideoCapture, VideoWriter};

use embedded_dl_with_rust::retinaface::{Detection, DetectionResult, RetinaFace};

const MODEL_PATH: &str = "tmp/retinaface_resnet50_trained_opt.trt";

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
    where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn main() {
    env_logger::init();

    let matches = App::new("")
        .arg(Arg::with_name("list_file")
            .long("list_file")
            .required(true)
            .takes_value(true))
        .arg(Arg::with_name("T")
            .long("T")
            .default_value("4")
            .takes_value(true))
        .arg(Arg::with_name("mul")
            .long("mul")
            .default_value("1")
            .takes_value(true))
        .get_matches();

    fs::remove_dir_all("./out").ok();
    fs::create_dir("./out").unwrap();

    let t = matches.value_of("T").unwrap().parse::<usize>().unwrap();
    let mul = matches.value_of("mul").unwrap().parse::<usize>().unwrap();
    let mut retinaface = RetinaFace::new(MODEL_PATH);

    let list_file = matches.value_of("list_file").unwrap();
    for line in read_lines(list_file).unwrap() {
        let fname = line.unwrap();
        let video_path = Path::new(&fname);
        detect(&mut retinaface, video_path, t, mul).unwrap();
    }
}

#[allow(unused)]
fn detect(retinaface: &mut RetinaFace, video_path: &Path, t: usize, mul: usize) -> Result<(), Error> {
    let mut cap = VideoCapture::new_from_file_with_backend(video_path.to_str().unwrap(), CAP_ANY)?;

    let codec = VideoWriter::fourcc('M' as i8, 'J' as i8, 'P' as i8, 'G' as i8)?;
    let fps = cap.get(CAP_PROP_FPS)?;
    let frame_count = cap.get(CAP_PROP_FRAME_COUNT)?;
    let frames_to_use = {
        let double = linspace::<f64>(0., frame_count, t * mul + 2)
            .map(|n| n as i64)
            .map(|n| [n, n + 1])
            .collect::<Vec<_>>();
        double[1..double.len() - 1].iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>()
    };

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

        let mut frame = Mat::default()?;
        if let Err(result) = cap.read(&mut frame) {
            break;
        }

        let detections = unsafe { retinaface.detect(&frame) };
        render(&mut frame, detections);
        let stem = video_path.file_stem().unwrap().to_str().unwrap();
        imwrite(&format!("out/{}-{:03}.jpg", stem, n), &frame, &VectorOfint::new())?;

        println!("{}", n);
    }
    let end = SystemTime::now();
    let elapsed = end.duration_since(start)?;
    println!("elapsed {:?}", elapsed);

    cap.release()?;
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
