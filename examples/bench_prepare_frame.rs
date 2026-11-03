//! Timing of `prepare_frame` on typical camera frames, next to what the surrounding copies cost.
//!
//! Run with: cargo run --release --example bench_prepare_frame

use std::hint::black_box;
use std::time::Instant;
use vitallens_core::image::{prepare_frame_into, Orientation, PixelFormat, PixelRect, Plane, YuvMatrix};

fn time_us(iters: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..iters / 10 + 1 {
        f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() * 1e6 / iters as f64
}

/// Per-pixel block averaging in the style of the current Kotlin `boxDownsample` (integer block
/// bounds, one pixel at a time), for comparison.
fn naive_box(data: &[u8], stride: usize, bpp: usize, crop: PixelRect, out: usize, dst: &mut [u8]) {
    let (cw, ch) = (crop.width as usize, crop.height as usize);
    for oy in 0..out {
        let (y0, y1) = (oy * ch / out, ((oy + 1) * ch / out).max(oy * ch / out + 1));
        for ox in 0..out {
            let (x0, x1) = (ox * cw / out, ((ox + 1) * cw / out).max(ox * cw / out + 1));
            let mut s = [0u32; 3];
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = (crop.y as usize + y) * stride + (crop.x as usize + x) * bpp;
                    s[0] += data[p] as u32;
                    s[1] += data[p + 1] as u32;
                    s[2] += data[p + 2] as u32;
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as u32;
            for c in 0..3 {
                dst[(oy * out + ox) * 3 + c] = ((s[c] + n / 2) / n) as u8;
            }
        }
    }
}

fn main() {
    let out = 40u32;
    let mut dst = vec![0u8; (out * out * 3) as usize];
    println!("{:<44} {:>10} {:>10} {:>12} {:>12}", "case", "core us", "naive us", "copy frame", "copy crop");
    let cases = [
        ("RGBA 640x480, face 300x300", PixelFormat::Rgba, 640u32, 480u32, 300u32),
        ("RGBA 1280x720, face 450x450", PixelFormat::Rgba, 1280, 720, 450),
        ("BGRA 1920x1080, face 700x700", PixelFormat::Bgra, 1920, 1080, 700),
        ("YUV420 NV12 640x480, face 300x300", PixelFormat::Yuv420, 640, 480, 300),
        ("YUV420 NV12 1280x720, face 450x450", PixelFormat::Yuv420, 1280, 720, 450),
        ("YUV420 NV12 1920x1080, face 700x700", PixelFormat::Yuv420, 1920, 1080, 700),
    ];
    let _ = time_us(2000, || { let v: Vec<u8> = vec![1u8; 1 << 20]; black_box(v); });
    for (name, fmt, w, h, face) in cases {
        let crop = PixelRect { x: (w - face) / 2 + 1, y: (h - face) / 2 + 1, width: face, height: face };
        let iters = 400;
        if fmt == PixelFormat::Yuv420 {
            let y: Vec<u8> = (0..w * h).map(|i| (i * 31 % 251) as u8).collect();
            let uv: Vec<u8> = (0..w * h / 2).map(|i| (i * 17 % 241) as u8).collect();
            let planes = [
                Plane { data: &y, row_stride: w as usize, pixel_stride: 1 },
                Plane { data: &uv[..uv.len() - 1], row_stride: w as usize, pixel_stride: 2 },
                Plane { data: &uv[1..], row_stride: w as usize, pixel_stride: 2 },
            ];
            let core = time_us(iters, || {
                prepare_frame_into(black_box(&planes), w, h, fmt, YuvMatrix::Bt601Full, crop, out, out, Orientation::UPRIGHT, &mut dst).unwrap();
            });
            let frame_bytes = y.len() + uv.len();
            let mut copy = vec![0u8; frame_bytes];
            let copy_frame = time_us(iters, || {
                copy[..y.len()].copy_from_slice(black_box(&y));
                copy[y.len()..].copy_from_slice(black_box(&uv));
            });
            println!("{:<44} {:>10.1} {:>10} {:>12.1} {:>12}", name, core, "-", copy_frame, "-");
        } else {
            let data: Vec<u8> = (0..w * h * 4).map(|i| (i * 31 % 251) as u8).collect();
            let plane = Plane { data: &data, row_stride: (w * 4) as usize, pixel_stride: 4 };
            let core = time_us(iters, || {
                prepare_frame_into(black_box(&[plane]), w, h, fmt, YuvMatrix::Bt601Full, crop, out, out, Orientation::UPRIGHT, &mut dst).unwrap();
            });
            let naive = time_us(iters, || naive_box(black_box(&data), (w * 4) as usize, 4, crop, out as usize, &mut dst));
            let mut copy = vec![0u8; data.len()];
            let copy_frame = time_us(iters, || copy.copy_from_slice(black_box(&data)));
            let crop_bytes = (face * 4) as usize;
            let mut crop_copy = vec![0u8; crop_bytes * face as usize];
            let copy_crop = time_us(iters, || {
                for r in 0..face as usize {
                    let s = (crop.y as usize + r) * (w * 4) as usize + crop.x as usize * 4;
                    crop_copy[r * crop_bytes..(r + 1) * crop_bytes].copy_from_slice(black_box(&data[s..s + crop_bytes]));
                }
            });
            println!("{:<44} {:>10.1} {:>10.1} {:>12.1} {:>12.1}", name, core, naive, copy_frame, copy_crop);
        }
    }
}
