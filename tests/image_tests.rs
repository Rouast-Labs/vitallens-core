//! Checks `prepare_frame` against a deliberately naive oracle: for each output pixel it visits
//! every source pixel of the frame, computes its exact covered area, and averages. Frames, strides,
//! crops, output sizes and orientations are randomised.

use vitallens_core::image::{
    prepare_frame, Orientation, PixelFormat, PixelRect, Plane, YuvMatrix,
};

/// Small deterministic PRNG (xorshift64*), so the tests need no extra dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % (hi - lo + 1) as u64) as u32
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 32) as u8
    }
}

/// Exact overlap, in units of 1/out source pixel, of source pixel `i` with output `o`.
fn overlap(i: u64, o: u64, start: u64, len: u64, out: u64) -> u64 {
    let (a, b) = (start * out + o * len, start * out + (o + 1) * len);
    let (p, q) = (i * out, (i + 1) * out);
    q.min(b).saturating_sub(p.max(a))
}

/// Rotates an RGB image 90 degrees clockwise.
fn rot90(img: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; img.len()];
    // New image is h wide, w tall; source (x, y) goes to (h - 1 - y, x).
    for y in 0..h {
        for x in 0..w {
            let (nx, ny) = (h - 1 - y, x);
            out[(ny * h + nx) * 3..(ny * h + nx) * 3 + 3].copy_from_slice(&img[(y * w + x) * 3..(y * w + x) * 3 + 3]);
        }
    }
    out
}

fn orient(mut img: Vec<u8>, mut w: usize, mut h: usize, o: Orientation) -> Vec<u8> {
    for _ in 0..o.rotation_degrees / 90 {
        img = rot90(&img, w, h);
        std::mem::swap(&mut w, &mut h);
    }
    if o.mirrored {
        for row in img.chunks_exact_mut(w * 3) {
            let px: Vec<[u8; 3]> = row.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).rev().collect();
            for (dst, p) in row.chunks_exact_mut(3).zip(px) {
                dst.copy_from_slice(&p);
            }
        }
    }
    img
}

/// Oracle: area-weighted mean of `sample(x, y) -> [f64; 3]` channels, then `finish` per pixel.
fn oracle(
    crop: PixelRect,
    out_w: u32,
    out_h: u32,
    o: Orientation,
    sample: impl Fn(u32, u32) -> [u64; 3],
    finish: impl Fn([u64; 3], u64) -> [u8; 3],
) -> Vec<u8> {
    let (tw, th) = if o.rotation_degrees % 180 == 90 { (out_h, out_w) } else { (out_w, out_h) };
    let total = crop.width as u64 * crop.height as u64;
    let mut raw = vec![0u8; (tw * th * 3) as usize];
    for oy in 0..th {
        for ox in 0..tw {
            let mut s = [0u64; 3];
            for y in crop.y..crop.y + crop.height {
                let wy = overlap(y as u64, oy as u64, crop.y as u64, crop.height as u64, th as u64);
                if wy == 0 { continue; }
                for x in crop.x..crop.x + crop.width {
                    let wx = overlap(x as u64, ox as u64, crop.x as u64, crop.width as u64, tw as u64);
                    if wx == 0 { continue; }
                    let v = sample(x, y);
                    for c in 0..3 {
                        s[c] += wx * wy * v[c];
                    }
                }
            }
            let px = finish(s, total);
            raw[((oy * tw + ox) * 3) as usize..((oy * tw + ox) * 3 + 3) as usize].copy_from_slice(&px);
        }
    }
    orient(raw, tw as usize, th as usize, o)
}

fn random_case(rng: &mut Rng) -> (u32, u32, PixelRect, u32, u32, Orientation) {
    let w = rng.range(2, 90);
    let h = rng.range(2, 90);
    let cw = rng.range(1, w);
    let ch = rng.range(1, h);
    let crop = PixelRect { x: rng.range(0, w - cw), y: rng.range(0, h - ch), width: cw, height: ch };
    let out_w = rng.range(1, 24);
    let out_h = if rng.range(0, 1) == 0 { out_w } else { rng.range(1, 24) };
    let o = Orientation { rotation_degrees: rng.range(0, 3) * 90, mirrored: rng.range(0, 1) == 1 };
    (w, h, crop, out_w, out_h, o)
}

#[test]
fn packed_formats_match_oracle() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let formats = [
        (PixelFormat::Rgb, 3usize, [0usize, 1, 2]),
        (PixelFormat::Bgr, 3, [2, 1, 0]),
        (PixelFormat::Rgba, 4, [0, 1, 2]),
        (PixelFormat::Bgra, 4, [2, 1, 0]),
        (PixelFormat::Argb, 4, [1, 2, 3]),
    ];
    for case in 0..400 {
        let (fmt, bpp, offs) = formats[case % formats.len()];
        let (w, h, crop, out_w, out_h, o) = random_case(&mut rng);
        let row_stride = w as usize * bpp + rng.range(0, 9) as usize;
        let data: Vec<u8> = (0..row_stride * h as usize).map(|_| rng.byte()).collect();
        let plane = Plane { data: &data, row_stride, pixel_stride: bpp };
        let got = prepare_frame(&[plane], w, h, fmt, YuvMatrix::Bt601Full, crop, out_w, out_h, o).unwrap();
        let want = oracle(
            crop, out_w, out_h, o,
            |x, y| {
                let p = y as usize * row_stride + x as usize * bpp;
                [data[p + offs[0]] as u64, data[p + offs[1]] as u64, data[p + offs[2]] as u64]
            },
            |s, t| [((s[0] + t / 2) / t) as u8, ((s[1] + t / 2) / t) as u8, ((s[2] + t / 2) / t) as u8],
        );
        assert_eq!(got, want, "case {case}: {fmt:?} {w}x{h} crop {crop:?} out {out_w}x{out_h} {o:?}");
    }
}

/// YUV layouts as Android delivers them: planar (I420) or interleaved chroma (NV12 / NV21).
#[derive(Clone, Copy, Debug)]
enum YuvLayout { I420, Nv12, Nv21 }

#[test]
fn yuv420_matches_oracle_for_planar_and_interleaved_chroma() {
    let mut rng = Rng(0xD1B54A32D192ED03);
    let matrices = [YuvMatrix::Bt601Limited, YuvMatrix::Bt601Full, YuvMatrix::Bt709Limited, YuvMatrix::Bt709Full];
    let layouts = [YuvLayout::I420, YuvLayout::Nv12, YuvLayout::Nv21];
    for case in 0..300 {
        let layout = layouts[case % 3];
        let matrix = matrices[(case / 3) % 4];
        let (w, h, crop, out_w, out_h, o) = random_case(&mut rng);
        let (cw, chh) = ((w as usize).div_ceil(2), (h as usize).div_ceil(2));
        let y_stride = w as usize + rng.range(0, 7) as usize;
        let y_plane: Vec<u8> = (0..y_stride * h as usize).map(|_| rng.byte()).collect();
        // Chroma sample values, then laid out per the layout.
        let u: Vec<u8> = (0..cw * chh).map(|_| rng.byte()).collect();
        let v: Vec<u8> = (0..cw * chh).map(|_| rng.byte()).collect();
        let (u_buf, v_buf, c_stride, c_ps, u_start, v_start);
        let interleaved: Vec<u8>;
        match layout {
            YuvLayout::I420 => {
                let s = cw + rng.range(0, 5) as usize;
                let mut ub = vec![0u8; s * chh];
                let mut vb = vec![0u8; s * chh];
                for r in 0..chh {
                    ub[r * s..r * s + cw].copy_from_slice(&u[r * cw..(r + 1) * cw]);
                    vb[r * s..r * s + cw].copy_from_slice(&v[r * cw..(r + 1) * cw]);
                }
                u_buf = ub; v_buf = vb; c_stride = s; c_ps = 1; u_start = 0; v_start = 0;
                interleaved = Vec::new();
            }
            YuvLayout::Nv12 | YuvLayout::Nv21 => {
                let s = cw * 2 + rng.range(0, 5) as usize;
                let mut buf = vec![0u8; s * chh];
                let (first, second) = if matches!(layout, YuvLayout::Nv12) { (&u, &v) } else { (&v, &u) };
                for r in 0..chh {
                    for c in 0..cw {
                        buf[r * s + c * 2] = first[r * cw + c];
                        buf[r * s + c * 2 + 1] = second[r * cw + c];
                    }
                }
                interleaved = buf;
                u_buf = Vec::new(); v_buf = Vec::new(); c_stride = s; c_ps = 2;
                (u_start, v_start) = if matches!(layout, YuvLayout::Nv12) { (0, 1) } else { (1, 0) };
            }
        }
        // Android's views of interleaved chroma: each starts at its first sample and ends right
        // after the last sample of the last row, one byte short of the other channel's end.
        let plane_end = |start: usize| start + (chh - 1) * c_stride + (cw - 1) * c_ps + 1;
        let (u_slice, v_slice): (&[u8], &[u8]) = match layout {
            YuvLayout::I420 => (&u_buf, &v_buf),
            _ => (
                &interleaved[u_start..plane_end(u_start)],
                &interleaved[v_start..plane_end(v_start)],
            ),
        };
        let planes = [
            Plane { data: &y_plane, row_stride: y_stride, pixel_stride: 1 },
            Plane { data: u_slice, row_stride: c_stride, pixel_stride: c_ps },
            Plane { data: v_slice, row_stride: c_stride, pixel_stride: c_ps },
        ];
        let got = prepare_frame(&planes, w, h, PixelFormat::Yuv420, matrix, crop, out_w, out_h, o).unwrap();

        let (yo, ky, rv, gu, gv, bu): (i64, i64, i64, i64, i64, i64) = match matrix {
            YuvMatrix::Bt601Limited => (16, 1192, 1634, 400, 833, 2066),
            YuvMatrix::Bt601Full => (0, 1024, 1436, 352, 731, 1815),
            YuvMatrix::Bt709Limited => (16, 1192, 1836, 218, 546, 2163),
            YuvMatrix::Bt709Full => (0, 1024, 1613, 192, 479, 1900),
        };
        let want = oracle(
            crop, out_w, out_h, o,
            |x, y| {
                let ci = (y as usize / 2) * cw + x as usize / 2;
                [y_plane[y as usize * y_stride + x as usize] as u64, u[ci] as u64, v[ci] as u64]
            },
            |s, t| {
                let t = t as i64;
                let (c, d, e) = (s[0] as i64 - yo * t, s[1] as i64 - 128 * t, s[2] as i64 - 128 * t);
                let den = 1024 * t;
                let r = |n: i64| (n + den / 2).div_euclid(den).clamp(0, 255) as u8;
                [r(ky * c + rv * e), r(ky * c - gu * d - gv * e), r(ky * c + bu * d)]
            },
        );
        assert_eq!(got, want, "case {case}: {layout:?} {matrix:?} {w}x{h} crop {crop:?} out {out_w}x{out_h} {o:?}");
    }
}

/// Averaging in YUV and converting once must agree with converting every pixel and averaging
/// after, for in-gamut colours (no clamping involved), up to rounding.
#[test]
fn averaging_before_conversion_equals_converting_each_pixel() {
    let mut rng = Rng(42);
    let (w, h) = (64u32, 48u32);
    // Mid-range values keep every pixel in gamut.
    let y: Vec<u8> = (0..w * h).map(|_| rng.range(90, 160) as u8).collect();
    let u: Vec<u8> = (0..(w / 2) * (h / 2)).map(|_| rng.range(110, 145) as u8).collect();
    let v: Vec<u8> = (0..(w / 2) * (h / 2)).map(|_| rng.range(110, 145) as u8).collect();
    let crop = PixelRect { x: 3, y: 5, width: 57, height: 41 };
    let planes = [
        Plane { data: &y, row_stride: w as usize, pixel_stride: 1 },
        Plane { data: &u, row_stride: (w / 2) as usize, pixel_stride: 1 },
        Plane { data: &v, row_stride: (w / 2) as usize, pixel_stride: 1 },
    ];
    let got = prepare_frame(&planes, w, h, PixelFormat::Yuv420, YuvMatrix::Bt601Limited, crop, 8, 8, Orientation::UPRIGHT).unwrap();
    // Convert per pixel in floating point, then area-average in floating point.
    let rgb_at = |x: u32, y_: u32| {
        let ci = ((y_ / 2) * (w / 2) + x / 2) as usize;
        let (yy, uu, vv) = (y[(y_ * w + x) as usize] as f64 - 16.0, u[ci] as f64 - 128.0, v[ci] as f64 - 128.0);
        [1.164 * yy + 1.596 * vv, 1.164 * yy - 0.391 * uu - 0.813 * vv, 1.164 * yy + 2.018 * uu]
    };
    for oy in 0..8u64 {
        for ox in 0..8u64 {
            let mut s = [0f64; 3];
            for yy in crop.y..crop.y + crop.height {
                for xx in crop.x..crop.x + crop.width {
                    let wgt = (overlap(xx as u64, ox, crop.x as u64, crop.width as u64, 8)
                        * overlap(yy as u64, oy, crop.y as u64, crop.height as u64, 8)) as f64;
                    let p = rgb_at(xx, yy);
                    for c in 0..3 { s[c] += wgt * p[c]; }
                }
            }
            let t = (crop.width * crop.height) as f64;
            for c in 0..3 {
                let want = s[c] / t;
                let have = got[((oy * 8 + ox) * 3) as usize + c] as f64;
                assert!((have - want).abs() <= 1.0, "pixel ({ox},{oy}) channel {c}: {have} vs {want:.2}");
            }
        }
    }
}

#[test]
fn rejects_crops_and_planes_that_do_not_fit() {
    let data = vec![0u8; 10 * 10 * 4];
    let plane = Plane { data: &data, row_stride: 40, pixel_stride: 4 };
    let call = |crop: PixelRect, p: Plane| prepare_frame(&[p], 10, 10, PixelFormat::Rgba, YuvMatrix::Bt601Full, crop, 4, 4, Orientation::UPRIGHT);
    assert!(call(PixelRect { x: 5, y: 0, width: 6, height: 4 }, plane).is_err());
    assert!(call(PixelRect { x: 0, y: 0, width: 0, height: 4 }, plane).is_err());
    let short = Plane { data: &data[..data.len() - 1], row_stride: 40, pixel_stride: 4 };
    // The last pixel's alpha byte is never read, so a buffer missing it is fine...
    assert!(call(PixelRect { x: 0, y: 0, width: 10, height: 10 }, short).is_ok());
    // ...but one missing a colour byte is rejected.
    let shorter = Plane { data: &data[..data.len() - 2], row_stride: 40, pixel_stride: 4 };
    assert!(call(PixelRect { x: 0, y: 0, width: 10, height: 10 }, shorter).is_err());
    assert!(prepare_frame(&[plane], 10, 10, PixelFormat::Rgba, YuvMatrix::Bt601Full,
        PixelRect { x: 0, y: 0, width: 4, height: 4 }, 4, 4, Orientation { rotation_degrees: 45, mirrored: false }).is_err());
}

/// The FFI entry points (by copy and by address) give the same bytes as the slice API.
#[test]
fn ffi_entry_points_match_slice_api() {
    use vitallens_core::image::{prepare_frame_from_addresses, prepare_frame_from_planes, FramePlane, FramePlaneAddress};
    let mut rng = Rng(7);
    let (w, h) = (64u32, 48u32);
    let y: Vec<u8> = (0..w * h).map(|_| rng.byte()).collect();
    let uv: Vec<u8> = (0..w * h / 2).map(|_| rng.byte()).collect();
    let crop = PixelRect { x: 3, y: 2, width: 51, height: 43 };
    let o = Orientation { rotation_degrees: 270, mirrored: true };
    let slices = [
        Plane { data: &y, row_stride: w as usize, pixel_stride: 1 },
        Plane { data: &uv[..uv.len() - 1], row_stride: w as usize, pixel_stride: 2 },
        Plane { data: &uv[1..], row_stride: w as usize, pixel_stride: 2 },
    ];
    let want = prepare_frame(&slices, w, h, PixelFormat::Yuv420, YuvMatrix::Bt601Full, crop, 10, 12, o).unwrap();
    let by_address: Vec<FramePlaneAddress> = slices
        .iter()
        .map(|p| FramePlaneAddress { address: p.data.as_ptr() as u64, length: p.data.len() as u64, row_stride: w, pixel_stride: p.pixel_stride as u32 })
        .collect();
    let by_copy: Vec<FramePlane> = slices
        .iter()
        .map(|p| FramePlane { data: p.data.to_vec(), row_stride: w, pixel_stride: p.pixel_stride as u32 })
        .collect();
    assert_eq!(prepare_frame_from_addresses(by_address, w, h, PixelFormat::Yuv420, YuvMatrix::Bt601Full, crop, 10, 12, o).unwrap(), want);
    assert_eq!(prepare_frame_from_planes(by_copy, w, h, PixelFormat::Yuv420, YuvMatrix::Bt601Full, crop, 10, 12, o).unwrap(), want);
    let null = vec![FramePlaneAddress { address: 0, length: 10, row_stride: 1, pixel_stride: 1 }; 3];
    assert!(prepare_frame_from_addresses(null, w, h, PixelFormat::Yuv420, YuvMatrix::Bt601Full, crop, 10, 12, o).is_err());
}
