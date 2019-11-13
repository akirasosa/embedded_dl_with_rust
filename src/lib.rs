#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::time::SystemTime;

include!("bindings.rs");

pub mod retinaface {
    use std::f32::consts::PI;
    use std::ffi::CString;
    use std::slice::from_raw_parts;

    use log;
    use opencv::core::{BORDER_CONSTANT, CV_32F, Mat, Point2f, Point2i, Rect, Scalar, Size, Size2i, split, transform};
    use opencv::dnn::nms_boxes;
    use opencv::imgproc::{INTER_LINEAR, warp_affine};
    use opencv::prelude::Vector;
    use opencv::Result;
    use opencv::types::{VectorOffloat, VectorOfint, VectorOfMat, VectorOfPoint2f, VectorOfRect};

    #[allow(unused_imports)]
    use crate::{bench, mal_Detection, mal_RetinaFace};

    const INPUT_SIZE: Size = Size { width: 512, height: 512 };
    const SCORE_THRE: f32 = 0.5;
    const NMS_THRE: f32 = 0.3;

    pub struct RetinaFace {
        retinaface: mal_RetinaFace
    }

    impl RetinaFace {
        pub fn new(model_path: &str) -> Self {
            let model_path = CString::new(model_path).unwrap();
            let ptr = model_path.as_ptr();
            let retinaface = unsafe { mal_RetinaFace::new(ptr) };
            RetinaFace { retinaface }
        }

        pub unsafe fn detect(&mut self, img: &Mat) -> DetectionResult {
//            let d = DetectionResult::from(&raw_detection).apply(nms);
//            bench(|| {
//                let (img, inv_mat) = wrap_in_blank(img, Size { width: 512, height: 512 }).unwrap();  // 0.47539094
//                let data = img_to_vec(&img).unwrap();  // 0.22908784
//                let raw_detection = self.retinaface.detect(data.as_ptr(), data.len());  // res50: 6.831405, mob_v2_25: 3.1893291
//                let d = DetectionResult::from(&raw_detection);  // 0.44168758
//                d.apply(nms);  // 0.04113181
//                d.transform(&inv_mat);  // 0.0043198904
//            });
            let (img, inv_mat) = wrap_in_blank(img, INPUT_SIZE).unwrap();
            let data = img_to_vec(&img).unwrap();
            let raw_detection = self.retinaface.detect(data.as_ptr(), data.len());

            DetectionResult::from(&raw_detection)
                .apply(nms)
                .transform(&inv_mat)
        }
    }

    impl Drop for RetinaFace {
        fn drop(&mut self) {
            unsafe { self.retinaface.destruct(); }
        }
    }

    pub struct DetectionResult {
        scores: Vec<f32>,
        points_n_7: Vec<Point2f>,
    }

    impl DetectionResult {
        fn from(raw: &mal_Detection) -> Self {
            let n_det = raw.nDet as usize;
            let scores = unsafe { from_raw_parts(raw.scores, n_det).to_vec() };
            let bboxes = unsafe { from_raw_parts(raw.bboxes, n_det * 4) };
            let landmarks = unsafe { from_raw_parts(raw.landmarks, n_det * 10) };

            let mut scores_new = vec![];
            let mut points_n_7 = vec![];

            scores.iter()
                .enumerate()
                .filter(|(_, &s)| s > SCORE_THRE)
                .for_each(|(i, &s)| {
                    scores_new.push(s);
                    let points_bb = (0..2)
                        .map(|j| {
                            Point2f { x: bboxes[i * 4 + j * 2], y: bboxes[i * 4 + j * 2 + 1] }
                        })
                        .collect::<Vec<Point2f>>();
                    let points_lm = (0..5)
                        .map(|j| {
                            Point2f { x: landmarks[i * 10 + j * 2], y: landmarks[i * 10 + j * 2 + 1] }
                        })
                        .collect::<Vec<Point2f>>();
                    points_n_7.extend(points_bb);
                    points_n_7.extend(points_lm);
                });
            log::debug!("After Score Thre: {} detections", scores_new.len());

            DetectionResult { scores: scores_new, points_n_7 }
        }

        pub fn n_det(&self) -> usize {
            self.scores.len()
        }

        fn bboxes_i(&self) -> Vec<Rect> {
            let n_det = self.n_det();
            (0..n_det)
                .map(|i| {
                    let tl: Point2i = self.points_n_7[i * 7 + 0].to().unwrap();
                    let br: Point2i = self.points_n_7[i * 7 + 1].to().unwrap();
                    Rect::from_points(tl, br)
                })
                .collect()
        }

        fn filter_by_indices(&self, indices: &Vec<usize>) -> Self {
            let scores = indices.iter()
                .map(|&idx| self.scores[idx])
                .collect();
            let points_n_7 = indices.iter()
                .flat_map(|&idx| {
                    self.points_n_7[idx * 7..(idx + 1) * 7].to_vec()
                })
                .collect();

            DetectionResult { scores, points_n_7 }
        }

        fn transform(&self, mat: &Mat) -> Self {
            let src = VectorOfPoint2f::from_iter(self.points_n_7.to_vec());
            let mut dst = VectorOfPoint2f::new();

            let (scores, points_n_7) = match transform(&src, &mut dst, &mat) {
                Ok(_) => (self.scores.to_vec(), dst.to_vec()),
                Err(e) => {
                    if self.points_n_7.len() != 0 {
                        log::warn!("Transform failed: {}", e);
                    }
                    (vec![], vec![])
                }
            };

            DetectionResult { scores, points_n_7 }
        }

        fn apply<F>(&self, func: F) -> Self where F: Fn(&Self) -> Self {
            func(self)
        }
    }

    impl IntoIterator for DetectionResult {
        type Item = Detection;
        type IntoIter = ::std::vec::IntoIter<Self::Item>;

        fn into_iter(self) -> Self::IntoIter {
            self.scores.iter()
                .enumerate()
                .map(|(i, &s)| {
                    Detection {
                        score: s,
                        bbox: {
                            let tl: Point2i = self.points_n_7[i * 7 + 0].to().unwrap();
                            let br: Point2i = self.points_n_7[i * 7 + 1].to().unwrap();
                            Rect::from_points(tl, br)
                        },
                        p0: self.points_n_7[i * 7 + 2].to().unwrap(),
                        p1: self.points_n_7[i * 7 + 3].to().unwrap(),
                        p2: self.points_n_7[i * 7 + 4].to().unwrap(),
                        p3: self.points_n_7[i * 7 + 5].to().unwrap(),
                        p4: self.points_n_7[i * 7 + 6].to().unwrap(),
                    }
                })
                .collect::<Vec<Detection>>()
                .into_iter()
        }
    }

    pub struct Detection {
        pub score: f32,
        pub bbox: Rect,
        pub p0: Point2i,
        pub p1: Point2i,
        pub p2: Point2i,
        pub p3: Point2i,
        pub p4: Point2i,
    }

    fn wrap_in_blank(img: &Mat, dst_size: Size2i) -> Result<(Mat, Mat)> {
        let src_size = img.size()?;
        let h = src_size.height as f32;
        let w = src_size.width as f32;
        let center = Point2f { x: w, y: h } * 0.5;
        let scale = h.max(w);
        let rot = 0.;

        let mat = get_affine_transform(center, scale, rot, dst_size, false)?;
        let mat_inv = get_affine_transform(center, scale, rot, dst_size, true)?;

        Mat::default().and_then(|mut dst| {
            warp_affine(&img, &mut dst, &mat, dst_size, INTER_LINEAR, BORDER_CONSTANT, Scalar::default())?;
            Ok((dst, mat_inv))
        })
    }

    fn img_to_vec(image: &Mat) -> Result<Vec<f32>> {
        let mut planes = VectorOfMat::new();
        planes.reserve(3);
        split(&image, &mut planes)?;

        let to_f32 = |src: Mat| {
            let mut dst = Mat::default()?;
            src.convert_to(&mut dst, CV_32F, 1., 0.)?;
            Ok(dst)
        };
        let r = planes.get(2).and_then(to_f32)?;
        let g = planes.get(1).and_then(to_f32)?;
        let b = planes.get(0).and_then(to_f32)?;

        Ok([r.data_typed()?, g.data_typed()?, b.data_typed()?].concat())
    }

    fn get_3rd_point(a: Point2f, b: Point2f) -> Point2f {
        let direct = a - b;
        b + Point2f { x: -direct.y, y: direct.x }
    }

    fn get_affine_transform(
        center: Point2f,
        scale: f32,
        rot: f32,
        dst_size: Size2i,
        inv: bool,
    ) -> Result<Mat> {
        let rot_rad = PI * rot as f32 / 180.;
        let src_point = Point2f { x: 0., y: scale * -0.5 };
        let src_dir = Point2f {
            x: src_point.x * rot_rad.cos() - src_point.y * rot_rad.sin(),
            y: src_point.x * rot_rad.sin() + src_point.y * rot_rad.cos(),
        };

        let dst_w = dst_size.width as f32;
        let dst_h = dst_size.height as f32;
        let dst_dir = Point2f { x: 0., y: dst_w * -0.5 };

        let src0 = center;
        let src1 = src0 + src_dir;
        let src2 = get_3rd_point(src0, src1);

        let dst0 = Point2f { x: dst_w, y: dst_h } * 0.5;
        let dst1 = dst0 + dst_dir;
        let dst2 = get_3rd_point(dst0, dst1);

        let src = VectorOfPoint2f::from_iter(vec![src0, src1, src2]);
        let dst = VectorOfPoint2f::from_iter(vec![dst0, dst1, dst2]);

        if inv {
            opencv::imgproc::get_affine_transform(&dst, &src)
        } else {
            opencv::imgproc::get_affine_transform(&src, &dst)
        }
    }

    fn nms(detection: &DetectionResult) -> DetectionResult {
        let scores = VectorOffloat::from_iter(detection.scores.to_vec());
        let bboxes = VectorOfRect::from_iter(detection.bboxes_i());

        let mut indices = VectorOfint::new();
        nms_boxes(&bboxes, &scores, SCORE_THRE, NMS_THRE, &mut indices, 1., 0).unwrap();

        let indices = indices.into_iter()
            .map(|i| i as usize)
            .collect::<Vec<usize>>();
        log::debug!("After NMS: {} detections", indices.len());

        detection.filter_by_indices(&indices)
    }
}

#[allow(unused)]
fn bench<F>(mut func: F) where F: FnMut() {
    // warm up
    for _ in 0..30 { func(); }

    let mut times = vec![];
    for _ in 0..100 {
        let start = SystemTime::now();
        func();
        let end = SystemTime::now();
        let elapsed = end.duration_since(start).unwrap();
        times.push(elapsed.as_nanos() as f32 / 1000. / 1000.);
    }
    let avg = times.into_iter().sum::<f32>() / 100.;

    println!("{} ms", avg);
}

