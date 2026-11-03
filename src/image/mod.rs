//! Turns a raw video frame into model input: crop, colour conversion, area downsampling,
//! rotation and mirroring in one pass, producing packed RGB bytes.
//!
//! The downsampling is an exact area (box) average: every output pixel is the mean of the source
//! pixels it covers, with partially covered pixels weighted by their covered fraction (the same
//! definition as OpenCV's `INTER_AREA`). Averaging over the whole area is what lifts the pulse
//! signal above sensor noise; point samplers such as bilinear discard it.
//!
//! All arithmetic is integer and exact, so every platform gets bit-identical output, and the
//! function serves as the reference for SDKs that keep a native path.
//!
//! Cost model (what makes it fast):
//! * Only the crop is read, each needed source row once per output row it touches.
//! * The vertical pass is a widening multiply-add of whole source rows into a column buffer,
//!   which the compiler vectorises; the horizontal pass then runs on the small column buffer.
//! * YUV frames are averaged in YUV and converted to RGB once per output pixel instead of once
//!   per source pixel. The conversion is affine, so this equals converting first and averaging
//!   after (chroma upsampled by repetition), apart from clamping of out-of-gamut pixels.
//!   Chroma planes are read at their own (subsampled) resolution.
//! * No per-frame conversion of the full frame and no intermediate image.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Pixel layout of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Enum))]
pub enum PixelFormat {
    /// One plane, 3 bytes per pixel: R, G, B.
    Rgb,
    /// One plane, 3 bytes per pixel: B, G, R.
    Bgr,
    /// One plane, 4 bytes per pixel: R, G, B, A (Android `RGBA_8888`, canvas `ImageData`).
    Rgba,
    /// One plane, 4 bytes per pixel: B, G, R, A (iOS `kCVPixelFormatType_32BGRA`).
    Bgra,
    /// One plane, 4 bytes per pixel: A, R, G, B.
    Argb,
    /// Three planes Y, U, V; chroma subsampled 2x2. Any row and pixel strides, so this covers
    /// I420 (planar), NV12 and NV21 (interleaved chroma, pixel stride 2), i.e. Android
    /// `YUV_420_888` and the planes of `MediaCodec` output images.
    Yuv420,
}

/// YUV to RGB conversion matrix and range, for [`PixelFormat::Yuv420`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Enum))]
pub enum YuvMatrix {
    /// BT.601, limited range (Y 16-235). Typical for decoded SD video.
    Bt601Limited,
    /// BT.601, full range (JFIF). Typical for Android camera `YUV_420_888`.
    Bt601Full,
    /// BT.709, limited range. Typical for decoded HD video.
    Bt709Limited,
    /// BT.709, full range.
    Bt709Full,
}

/// A rectangle in pixel coordinates of the source frame (raw, unrotated orientation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Rotation and mirroring applied after downsampling, to bring the raw frame upright.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct Orientation {
    /// Clockwise rotation in degrees: 0, 90, 180 or 270.
    pub rotation_degrees: u32,
    /// Mirror horizontally after rotating.
    pub mirrored: bool,
}

impl Orientation {
    pub const UPRIGHT: Orientation = Orientation { rotation_degrees: 0, mirrored: false };
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Error))]
#[cfg_attr(not(target_arch = "wasm32"), uniffi(flat_error))]
pub enum FrameError {
    #[error("invalid frame input: {0}")]
    InvalidInput(String),
}

fn invalid(msg: impl Into<String>) -> FrameError {
    FrameError::InvalidInput(msg.into())
}

/// One plane of a frame in memory.
#[derive(Debug, Clone, Copy)]
pub struct Plane<'a> {
    pub data: &'a [u8],
    /// Bytes from the start of one row to the start of the next.
    pub row_stride: usize,
    /// Bytes from one pixel (or chroma sample) to the next within a row.
    pub pixel_stride: usize,
}

/// Width and height of the raw output before rotation.
fn raw_output_size(out_width: u32, out_height: u32, orientation: Orientation) -> (u32, u32) {
    if orientation.rotation_degrees % 180 == 90 { (out_height, out_width) } else { (out_width, out_height) }
}

/// Crops `crop` out of the frame, area-averages it to `out_width` x `out_height` (the size after
/// rotation), converts to RGB, applies `orientation`, and writes packed RGB bytes into `out`
/// (`out_width * out_height * 3` bytes, row-major).
///
/// `planes` holds one plane for packed formats and Y, U, V for [`PixelFormat::Yuv420`].
/// `width`/`height` are the frame's luma size; `yuv_matrix` is ignored for packed formats.
#[allow(clippy::too_many_arguments)]
pub fn prepare_frame_into(
    planes: &[Plane<'_>],
    width: u32,
    height: u32,
    format: PixelFormat,
    yuv_matrix: YuvMatrix,
    crop: PixelRect,
    out_width: u32,
    out_height: u32,
    orientation: Orientation,
    out: &mut [u8],
) -> Result<(), FrameError> {
    if crop.width == 0 || crop.height == 0 {
        return Err(invalid("crop must not be empty"));
    }
    if out_width == 0 || out_height == 0 {
        return Err(invalid("output size must not be zero"));
    }
    if crop.x as u64 + crop.width as u64 > width as u64 || crop.y as u64 + crop.height as u64 > height as u64 {
        return Err(invalid(format!(
            "crop {}x{} at ({}, {}) is outside the {}x{} frame",
            crop.width, crop.height, crop.x, crop.y, width, height
        )));
    }
    if !matches!(orientation.rotation_degrees, 0 | 90 | 180 | 270) {
        return Err(invalid("rotation must be 0, 90, 180 or 270 degrees"));
    }
    let n_out = out_width as usize * out_height as usize;
    if out.len() != n_out * 3 {
        return Err(invalid(format!("output buffer must hold {} bytes", n_out * 3)));
    }

    let (tw, th) = raw_output_size(out_width, out_height, orientation);
    // Per output pixel, the sum of each channel weighted by covered area. Total weight per output
    // pixel is crop.width * crop.height in these units.
    let mut acc = vec![0u64; n_out * 3];

    match format {
        PixelFormat::Yuv420 => {
            if planes.len() != 3 {
                return Err(invalid("Yuv420 needs three planes (Y, U, V)"));
            }
            for (i, plane) in planes.iter().enumerate() {
                let shift = if i == 0 { 0 } else { 1 };
                accumulate_plane(plane, shift, &[(0, i)], crop, tw, th, &mut acc)?;
            }
            let total = crop.width as u64 * crop.height as u64;
            let coeffs = YuvCoeffs::for_matrix(yuv_matrix);
            let mut rgb = vec![0u8; n_out * 3];
            for (px, sums) in rgb.chunks_exact_mut(3).zip(acc.chunks_exact(3)) {
                let [r, g, b] = coeffs.to_rgb(sums[0], sums[1], sums[2], total);
                px[0] = r;
                px[1] = g;
                px[2] = b;
            }
            write_oriented(&rgb, tw, th, orientation, out);
        }
        _ => {
            if planes.len() != 1 {
                return Err(invalid("packed formats need exactly one plane"));
            }
            let (bpp, offsets) = packed_layout(format);
            let plane = planes[0];
            if plane.pixel_stride < bpp {
                return Err(invalid(format!("pixel stride must be at least {bpp} for {format:?}")));
            }
            let channels = [(offsets[0], 0), (offsets[1], 1), (offsets[2], 2)];
            accumulate_plane(&plane, 0, &channels, crop, tw, th, &mut acc)?;
            let total = crop.width as u64 * crop.height as u64;
            let half = total / 2;
            let mut rgb = vec![0u8; n_out * 3];
            for (v, s) in rgb.iter_mut().zip(acc.iter()) {
                *v = ((s + half) / total) as u8;
            }
            write_oriented(&rgb, tw, th, orientation, out);
        }
    }
    Ok(())
}

/// Convenience wrapper around [`prepare_frame_into`] that allocates the output.
#[allow(clippy::too_many_arguments)]
pub fn prepare_frame(
    planes: &[Plane<'_>],
    width: u32,
    height: u32,
    format: PixelFormat,
    yuv_matrix: YuvMatrix,
    crop: PixelRect,
    out_width: u32,
    out_height: u32,
    orientation: Orientation,
) -> Result<Vec<u8>, FrameError> {
    let mut out = vec![0u8; out_width as usize * out_height as usize * 3];
    prepare_frame_into(planes, width, height, format, yuv_matrix, crop, out_width, out_height, orientation, &mut out)?;
    Ok(out)
}

/// Bytes per pixel and the R, G, B byte offsets of a packed format.
fn packed_layout(format: PixelFormat) -> (usize, [usize; 3]) {
    match format {
        PixelFormat::Rgb => (3, [0, 1, 2]),
        PixelFormat::Bgr => (3, [2, 1, 0]),
        PixelFormat::Rgba => (4, [0, 1, 2]),
        PixelFormat::Bgra => (4, [2, 1, 0]),
        PixelFormat::Argb => (4, [1, 2, 3]),
        PixelFormat::Yuv420 => unreachable!("not a packed format"),
    }
}

/// The source samples one output pixel covers along one axis, with their weights.
///
/// Weights are exact overlaps in units of 1/out_len source pixel: the first and last samples may
/// be partially covered, all in between are fully covered (weight `mid`). Weights sum to
/// `crop_len`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    first: usize,
    count: usize,
    w_first: u32,
    w_mid: u32,
    w_last: u32,
}

/// Spans for each of `out_len` outputs along an axis, for a plane subsampled by `1 << shift`
/// relative to the luma grid on which the crop is given.
fn axis_spans(crop_start: u32, crop_len: u32, out_len: u32, shift: u32) -> Vec<Span> {
    let (start, len, n) = (crop_start as u64, crop_len as u64, out_len as u64);
    // One plane sample covers (1 << shift) luma pixels, i.e. `sample` units.
    let sample = n << shift;
    (0..n)
        .map(|o| {
            let a = start * n + o * len;
            let b = a + len;
            let first = a / sample;
            let end = b.div_ceil(sample);
            let weight = |k: u64| (b.min((k + 1) * sample) - a.max(k * sample)) as u32;
            let count = end - first;
            Span {
                first: first as usize,
                count: count as usize,
                w_first: weight(first),
                w_mid: sample as u32,
                w_last: if count > 1 { weight(end - 1) } else { 0 },
            }
        })
        .collect()
}

/// Adds one plane's area-weighted sums into `acc`.
///
/// `channels` lists (byte offset within a pixel, channel index in `acc`). The vertical pass
/// accumulates whole rows of the crop into a column buffer (a widening multiply-add over
/// contiguous bytes); the horizontal pass then reduces the column buffer per output pixel.
fn accumulate_plane(
    plane: &Plane<'_>,
    shift: u32,
    channels: &[(usize, usize)],
    crop: PixelRect,
    tw: u32,
    th: u32,
    acc: &mut [u64],
) -> Result<(), FrameError> {
    let ps = plane.pixel_stride;
    let rs = plane.row_stride;
    if ps == 0 {
        return Err(invalid("pixel stride must not be zero"));
    }
    let xs = axis_spans(crop.x, crop.width, tw, shift);
    let ys = axis_spans(crop.y, crop.height, th, shift);

    let col_first = xs[0].first;
    let col_last = xs.last().map(|s| s.first + s.count - 1).unwrap();
    let row_last = ys.last().map(|s| s.first + s.count - 1).unwrap();
    let max_off = channels.iter().map(|c| c.0).max().unwrap();
    // Bytes per row the vertical pass reads. It stops at the last needed byte rather than at the
    // end of the last pixel: Android's interleaved chroma planes end one byte early.
    let row_bytes = (col_last - col_first) * ps + max_off + 1;

    let last_byte = row_last
        .checked_mul(rs)
        .and_then(|v| v.checked_add(col_first * ps + row_bytes))
        .ok_or_else(|| invalid("plane size overflows"))?;
    if last_byte > plane.data.len() {
        return Err(invalid(format!(
            "plane holds {} bytes, but the crop needs {}",
            plane.data.len(),
            last_byte
        )));
    }

    // Padded to whole pixels so the horizontal pass can walk it pixel by pixel; the padding
    // beyond `row_bytes` stays zero and is never part of a used channel.
    let mut cols = vec![0u32; (col_last - col_first + 1) * ps];
    let mut mid = vec![0u16; row_bytes];
    let row_at = |r: usize| {
        let start = r * rs + col_first * ps;
        &plane.data[start..start + row_bytes]
    };
    let n_ch = 3;
    for (oy, ys) in ys.iter().enumerate() {
        // Edge rows carry their own weights; fully covered rows all weigh `w_mid`, so they are
        // summed unweighted in 16 bits (twice the SIMD lanes of 32 bits) and weighted once.
        cols.fill(0);
        for (c, &v) in cols.iter_mut().zip(row_at(ys.first)) {
            *c = ys.w_first * u32::from(v);
        }
        if ys.count > 1 {
            for (c, &v) in cols.iter_mut().zip(row_at(ys.first + ys.count - 1)) {
                *c += ys.w_last * u32::from(v);
            }
        }
        // 257 rows of 255 still fit in u16; flush before that.
        let mut r = 1;
        while r + 1 < ys.count {
            let batch_end = (r + 257).min(ys.count - 1);
            mid.fill(0);
            for row in r..batch_end {
                for (m, &v) in mid.iter_mut().zip(row_at(ys.first + row)) {
                    *m += u16::from(v);
                }
            }
            for (c, &m) in cols.iter_mut().zip(mid.iter()) {
                *c += ys.w_mid * u32::from(m);
            }
            r = batch_end;
        }
        let out_row = &mut acc[oy * tw as usize * n_ch..(oy + 1) * tw as usize * n_ch];
        match *channels {
            [(o0, c0)] => horizontal_pass(&cols, &xs, col_first, ps, [o0], [c0], out_row),
            [(o0, c0), (o1, c1), (o2, c2)] => {
                horizontal_pass(&cols, &xs, col_first, ps, [o0, o1, o2], [c0, c1, c2], out_row)
            }
            _ => unreachable!("one or three channels per plane"),
        }
    }
    Ok(())
}

/// Reduces one row of column sums to output pixels: for each output, the partially covered first
/// and last columns with their weights plus the fully covered ones with the shared weight, all
/// `N` channels in one walk over the output's columns.
fn horizontal_pass<const N: usize>(
    cols: &[u32],
    xs: &[Span],
    col_first: usize,
    ps: usize,
    offs: [usize; N],
    chs: [usize; N],
    out_row: &mut [u64],
) {
    for (ox, s) in xs.iter().enumerate() {
        let start = (s.first - col_first) * ps;
        let seg = &cols[start..start + s.count * ps];
        let first = &seg[..ps];
        let mut sums = [0u64; N];
        for j in 0..N {
            sums[j] = u64::from(s.w_first) * u64::from(first[offs[j]]);
        }
        if s.count > 1 {
            let last = &seg[(s.count - 1) * ps..];
            let inner = &seg[ps..(s.count - 1) * ps];
            let mut mids = [0u64; N];
            if N == 1 {
                // One channel: a plain (strided) sum, which vectorises for pixel stride 1.
                mids[0] = if ps == 1 {
                    inner.iter().map(|&v| u64::from(v)).sum()
                } else {
                    inner[offs[0]..].iter().step_by(ps).map(|&v| u64::from(v)).sum()
                };
            } else {
                for px in inner.chunks_exact(ps) {
                    for j in 0..N {
                        mids[j] += u64::from(px[offs[j]]);
                    }
                }
            }
            for j in 0..N {
                sums[j] += u64::from(s.w_last) * u64::from(last[offs[j]]) + u64::from(s.w_mid) * mids[j];
            }
        }
        for j in 0..N {
            out_row[ox * 3 + chs[j]] += sums[j];
        }
    }
}

/// Fixed-point YUV to RGB coefficients, scaled by 1024.
struct YuvCoeffs {
    y_offset: i64,
    y: i64,
    rv: i64,
    gu: i64,
    gv: i64,
    bu: i64,
}

impl YuvCoeffs {
    fn for_matrix(m: YuvMatrix) -> Self {
        match m {
            YuvMatrix::Bt601Limited => YuvCoeffs { y_offset: 16, y: 1192, rv: 1634, gu: 400, gv: 833, bu: 2066 },
            YuvMatrix::Bt601Full => YuvCoeffs { y_offset: 0, y: 1024, rv: 1436, gu: 352, gv: 731, bu: 1815 },
            YuvMatrix::Bt709Limited => YuvCoeffs { y_offset: 16, y: 1192, rv: 1836, gu: 218, gv: 546, bu: 2163 },
            YuvMatrix::Bt709Full => YuvCoeffs { y_offset: 0, y: 1024, rv: 1613, gu: 192, gv: 479, bu: 1900 },
        }
    }

    /// Converts area-weighted sums (each over `total` weight) to rounded, clamped RGB.
    fn to_rgb(&self, sy: u64, su: u64, sv: u64, total: u64) -> [u8; 3] {
        let t = total as i64;
        let c = sy as i64 - self.y_offset * t;
        let d = su as i64 - 128 * t;
        let e = sv as i64 - 128 * t;
        let denom = 1024 * t;
        let round = |num: i64| (num + denom / 2).div_euclid(denom).clamp(0, 255) as u8;
        [
            round(self.y * c + self.rv * e),
            round(self.y * c - self.gu * d - self.gv * e),
            round(self.y * c + self.bu * d),
        ]
    }
}

/// Writes the raw `tw` x `th` RGB image into `out`, rotated clockwise and then mirrored.
fn write_oriented(rgb: &[u8], tw: u32, th: u32, orientation: Orientation, out: &mut [u8]) {
    let (w, h) = (tw as usize, th as usize);
    let turns = orientation.rotation_degrees / 90;
    let new_w = if turns % 2 == 1 { h } else { w };
    for y in 0..h {
        for x in 0..w {
            let (mut nx, ny) = match turns {
                1 => (h - 1 - y, x),
                2 => (w - 1 - x, h - 1 - y),
                3 => (y, w - 1 - x),
                _ => (x, y),
            };
            if orientation.mirrored {
                nx = new_w - 1 - nx;
            }
            let src = (y * w + x) * 3;
            let dst = (ny * new_w + nx) * 3;
            out[dst..dst + 3].copy_from_slice(&rgb[src..src + 3]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Plain C entry point (Kotlin per-frame fast path)
// ---------------------------------------------------------------------------------------------

/// Numeric codes of [`PixelFormat`] and [`YuvMatrix`] for the C entry point.
#[cfg(not(target_arch = "wasm32"))]
fn format_from_code(code: u32) -> Option<PixelFormat> {
    Some(match code {
        0 => PixelFormat::Rgb,
        1 => PixelFormat::Bgr,
        2 => PixelFormat::Rgba,
        3 => PixelFormat::Bgra,
        4 => PixelFormat::Argb,
        5 => PixelFormat::Yuv420,
        _ => return None,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn matrix_from_code(code: u32) -> Option<YuvMatrix> {
    Some(match code {
        0 => YuvMatrix::Bt601Limited,
        1 => YuvMatrix::Bt601Full,
        2 => YuvMatrix::Bt709Limited,
        3 => YuvMatrix::Bt709Full,
        _ => return None,
    })
}

/// `prepare_frame` with only scalar arguments, writing into `out`, for callers whose FFI
/// marshals records slowly. UniFFI's Kotlin bindings pass every record through JNA structures,
/// which costs about a millisecond per call on Android, more than the work itself; the
/// hand-written Kotlin wrapper calls this through JNA direct mapping instead.
///
/// Planes are (address, length, row stride, pixel stride); packed formats use only the first,
/// and the other addresses may be 0. Formats: 0 Rgb, 1 Bgr, 2 Rgba, 3 Bgra, 4 Argb, 5 Yuv420.
/// Matrices: 0 Bt601Limited, 1 Bt601Full, 2 Bt709Limited, 3 Bt709Full.
///
/// Returns 0 on success. On failure returns -1 and writes a UTF-8 message (truncated, not
/// terminated) into `err`, its length into `err_len`.
///
/// # Safety
/// Every non-null address must point to at least its length in readable bytes, `out` to
/// `out_len` writable bytes and `err` to `err_cap` writable bytes, all valid for the call.
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn vitallens_prepare_frame(
    p0: u64, p0_len: u64, p0_row_stride: u32, p0_pixel_stride: u32,
    p1: u64, p1_len: u64, p1_row_stride: u32, p1_pixel_stride: u32,
    p2: u64, p2_len: u64, p2_row_stride: u32, p2_pixel_stride: u32,
    width: u32, height: u32, format: u32, yuv_matrix: u32,
    crop_x: u32, crop_y: u32, crop_width: u32, crop_height: u32,
    out_width: u32, out_height: u32, rotation_degrees: u32, mirrored: u32,
    out: u64, out_len: u64,
    err: u64, err_cap: u32, err_len: u64,
) -> i32 {
    let result = (|| {
        let format = format_from_code(format).ok_or_else(|| invalid("unknown pixel format code"))?;
        let matrix = matrix_from_code(yuv_matrix).ok_or_else(|| invalid("unknown YUV matrix code"))?;
        let n_planes = if format == PixelFormat::Yuv420 { 3 } else { 1 };
        let raw = [
            (p0, p0_len, p0_row_stride, p0_pixel_stride),
            (p1, p1_len, p1_row_stride, p1_pixel_stride),
            (p2, p2_len, p2_row_stride, p2_pixel_stride),
        ];
        let mut planes = Vec::with_capacity(n_planes);
        for &(addr, len, rs, ps) in &raw[..n_planes] {
            if addr == 0 {
                return Err(invalid("plane address is null"));
            }
            let len = usize::try_from(len).map_err(|_| invalid("plane length too large"))?;
            // SAFETY: the caller guarantees `len` readable bytes at `addr` for the call.
            let data = unsafe { std::slice::from_raw_parts(addr as usize as *const u8, len) };
            planes.push(Plane { data, row_stride: rs as usize, pixel_stride: ps as usize });
        }
        if out == 0 {
            return Err(invalid("output address is null"));
        }
        let out_len = usize::try_from(out_len).map_err(|_| invalid("output length too large"))?;
        // SAFETY: the caller guarantees `out_len` writable bytes at `out` for the call.
        let out = unsafe { std::slice::from_raw_parts_mut(out as usize as *mut u8, out_len) };
        prepare_frame_into(
            &planes, width, height, format, matrix,
            PixelRect { x: crop_x, y: crop_y, width: crop_width, height: crop_height },
            out_width, out_height,
            Orientation { rotation_degrees, mirrored: mirrored != 0 },
            out,
        )
    })();
    match result {
        Ok(()) => 0,
        Err(e) => {
            if err != 0 && err_len != 0 {
                let msg = e.to_string();
                let n = msg.len().min(err_cap as usize);
                // SAFETY: the caller guarantees `err_cap` writable bytes at `err` and a u32 at `err_len`.
                unsafe {
                    std::ptr::copy_nonoverlapping(msg.as_ptr(), err as usize as *mut u8, n);
                    *(err_len as usize as *mut u32) = n as u32;
                }
            }
            -1
        }
    }
}

// ---------------------------------------------------------------------------------------------
// UniFFI (Kotlin, Swift)
// ---------------------------------------------------------------------------------------------

/// A frame plane passed by copy.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, uniffi::Record)]
pub struct FramePlane {
    pub data: Vec<u8>,
    pub row_stride: u32,
    pub pixel_stride: u32,
}

/// A frame plane passed by address, so the frame is read in place without copying it across
/// the FFI. The caller must keep `length` bytes at `address` valid and unmodified for the
/// duration of the call (e.g. an open `ImageProxy`/`Image`, a locked `CVPixelBuffer`).
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, uniffi::Record)]
pub struct FramePlaneAddress {
    pub address: u64,
    pub length: u64,
    pub row_stride: u32,
    pub pixel_stride: u32,
}

/// Crops, area-averages, converts and orients a frame to packed RGB model input
/// (`out_width * out_height * 3` bytes). Planes are copied across the FFI; for camera frames
/// prefer [`prepare_frame_from_addresses`].
#[cfg(not(target_arch = "wasm32"))]
#[uniffi::export]
#[allow(clippy::too_many_arguments)]
pub fn prepare_frame_from_planes(
    planes: Vec<FramePlane>,
    width: u32,
    height: u32,
    format: PixelFormat,
    yuv_matrix: YuvMatrix,
    crop: PixelRect,
    out_width: u32,
    out_height: u32,
    orientation: Orientation,
) -> Result<Vec<u8>, FrameError> {
    let planes: Vec<Plane<'_>> = planes
        .iter()
        .map(|p| Plane { data: &p.data, row_stride: p.row_stride as usize, pixel_stride: p.pixel_stride as usize })
        .collect();
    prepare_frame(&planes, width, height, format, yuv_matrix, crop, out_width, out_height, orientation)
}

/// Same as [`prepare_frame_from_planes`], but reads the planes in place from native memory
/// (zero copy). Each address must point to `length` readable bytes that stay valid and
/// unchanged until the call returns; only bytes within that length are read.
#[cfg(not(target_arch = "wasm32"))]
#[uniffi::export]
#[allow(clippy::too_many_arguments)]
pub fn prepare_frame_from_addresses(
    planes: Vec<FramePlaneAddress>,
    width: u32,
    height: u32,
    format: PixelFormat,
    yuv_matrix: YuvMatrix,
    crop: PixelRect,
    out_width: u32,
    out_height: u32,
    orientation: Orientation,
) -> Result<Vec<u8>, FrameError> {
    let mut slices = Vec::with_capacity(planes.len());
    for p in &planes {
        if p.address == 0 {
            return Err(invalid("plane address is null"));
        }
        let len = usize::try_from(p.length).map_err(|_| invalid("plane length too large"))?;
        // SAFETY: the caller guarantees `len` readable bytes at `address` for the duration of
        // this call; bounds are checked against `len` before any read.
        let data = unsafe { std::slice::from_raw_parts(p.address as usize as *const u8, len) };
        slices.push(Plane { data, row_stride: p.row_stride as usize, pixel_stride: p.pixel_stride as usize });
    }
    prepare_frame(&slices, width, height, format, yuv_matrix, crop, out_width, out_height, orientation)
}

// ---------------------------------------------------------------------------------------------
// Wasm (JS)
// ---------------------------------------------------------------------------------------------

/// Packed-format frames for JS (`format`: "rgb", "bgr", "rgba", "bgra", "argb"), e.g. canvas
/// `ImageData` as "rgba" with `rowStride = width * 4`. The bytes are copied into wasm memory once.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = prepareFrame)]
#[allow(clippy::too_many_arguments)]
pub fn prepare_frame_js(
    data: &[u8],
    width: u32,
    height: u32,
    row_stride: u32,
    format: &str,
    crop_x: u32,
    crop_y: u32,
    crop_width: u32,
    crop_height: u32,
    out_width: u32,
    out_height: u32,
    rotation_degrees: u32,
    mirrored: bool,
) -> Result<Vec<u8>, JsError> {
    let format = parse_packed_format(format).map_err(|e| JsError::new(&e.to_string()))?;
    let (bpp, _) = packed_layout(format);
    let plane = Plane { data, row_stride: row_stride as usize, pixel_stride: bpp };
    prepare_frame(
        &[plane],
        width,
        height,
        format,
        YuvMatrix::Bt601Full,
        PixelRect { x: crop_x, y: crop_y, width: crop_width, height: crop_height },
        out_width,
        out_height,
        Orientation { rotation_degrees, mirrored },
    )
    .map_err(|e| JsError::new(&e.to_string()))
}

/// Parses a packed pixel format name ("rgb", "bgr", "rgba", "bgra", "argb").
pub fn parse_packed_format(name: &str) -> Result<PixelFormat, FrameError> {
    match name.to_ascii_lowercase().as_str() {
        "rgb" => Ok(PixelFormat::Rgb),
        "bgr" => Ok(PixelFormat::Bgr),
        "rgba" => Ok(PixelFormat::Rgba),
        "bgra" => Ok(PixelFormat::Bgra),
        "argb" => Ok(PixelFormat::Argb),
        other => Err(invalid(format!("unknown packed pixel format '{other}'"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_cover_exactly_and_weights_sum_to_crop_len() {
        for &(start, len, out, shift) in &[(0, 300, 40, 0), (7, 301, 40, 0), (3, 37, 40, 0), (5, 123, 40, 1), (0, 1, 3, 0)] {
            let spans = axis_spans(start, len, out, shift);
            for s in &spans {
                let mid = s.count.saturating_sub(2) as u32 * s.w_mid;
                let total = s.w_first + if s.count > 1 { s.w_last } else { 0 } + mid;
                assert_eq!(total, len, "start {start} len {len} out {out} shift {shift}: {s:?}");
            }
        }
    }

    #[test]
    fn integer_blocks_match_plain_block_means() {
        // 4x4 RGB, crop all, out 2x2: each output is the rounded mean of a 2x2 block.
        let mut data = vec![0u8; 4 * 4 * 3];
        for (i, v) in data.iter_mut().enumerate() {
            *v = (i * 7 % 256) as u8;
        }
        let plane = Plane { data: &data, row_stride: 12, pixel_stride: 3 };
        let out = prepare_frame(&[plane], 4, 4, PixelFormat::Rgb, YuvMatrix::Bt601Full,
            PixelRect { x: 0, y: 0, width: 4, height: 4 }, 2, 2, Orientation::UPRIGHT).unwrap();
        for oy in 0..2 {
            for ox in 0..2 {
                for c in 0..3 {
                    let mut s = 0u32;
                    for y in 0..2 {
                        for x in 0..2 {
                            s += data[((oy * 2 + y) * 4 + ox * 2 + x) * 3 + c] as u32;
                        }
                    }
                    assert_eq!(out[(oy * 2 + ox) * 3 + c] as u32, (s + 2) / 4);
                }
            }
        }
    }
}
