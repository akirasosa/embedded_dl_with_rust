use opencv::core::{Point, Scalar, ToInputOutputArray, Rect, Mat};
use opencv::imgcodecs::{imread, IMREAD_COLOR, imwrite};
use opencv::imgproc::LINE_8;
use opencv::prelude::Vector;
use opencv::Result;
use opencv::types::VectorOfint;

use opencv::videoio::{VideoCapture, CAP_ANY, VideoWriter, CAP_PROP_FPS};
use embedded_dl_with_rust::retinaface::{RetinaFace, DetectionResult, Detection};

const MODEL_PATH: &str = "tmp/retinaface_resnet50_trained_opt.trt";

fn main() {
    env_logger::init();
    run().unwrap();
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
