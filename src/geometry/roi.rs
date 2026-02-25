use crate::types::{Rect, RoiMethod, FaceDetector};

/// Standard Face Crop (Reduces width to 60%, height to 80%)
const FACE_OFFSETS: [f32; 4] = [-0.2, -0.1, -0.2, -0.1];

/// Forehead Crop (Top 15-25% of face)
const FOREHEAD_OFFSETS: [f32; 4] = [-0.35, -0.15, -0.35, -0.75];

/// Standard Upper Body (Version 1 - Non-Cropped)
const UPPER_BODY_OFFSETS: [f32; 4] = [0.25, 0.20, 0.25, 0.40];

/// Upper Body Cropped (Version 1 - Cropped)
const UPPER_BODY_CROPPED_OFFSETS: [f32; 4] = [0.19, 0.1455, 0.19, 0.2769];

/// Offsets to convert other Face Detectors to default
const VISION_TO_DEFAULT_OFFSETS: [f32; 4] = [-0.05, 0.20, -0.05, 0.25];

#[cfg_attr(not(target_arch = "wasm32"), uniffi::export)]
pub fn calculate_roi(
    face: Rect,
    method: RoiMethod,
    detector: FaceDetector,
    container_width: Option<f32>,
    container_height: Option<f32>,
    force_even: bool
) -> Rect {
    let base_face = normalize_face_rect(face, detector);

    let offsets = match method {
        RoiMethod::Face => FACE_OFFSETS,
        RoiMethod::Forehead => FOREHEAD_OFFSETS,
        RoiMethod::UpperBody => UPPER_BODY_OFFSETS,
        RoiMethod::UpperBodyCropped => UPPER_BODY_CROPPED_OFFSETS,
        RoiMethod::Custom { left, top, right, bottom } => [left, top, right, bottom],
    };

    let container = match (container_width, container_height) {
        (Some(w), Some(h)) => Some((w, h)),
        _ => None
    };

    apply_offsets(base_face, offsets, container, force_even)
}

fn apply_offsets(
    rect: Rect, 
    offsets: [f32; 4], 
    container: Option<(f32, f32)>,
    force_even: bool
) -> Rect {
    let [ch_left, ch_top, ch_right, ch_bottom] = offsets;

    let abs_ch_left = ch_left * rect.width;
    let abs_ch_top = ch_top * rect.height;
    let abs_ch_right = ch_right * rect.width;
    let abs_ch_bottom = ch_bottom * rect.height;

    let mut new_x = rect.x - abs_ch_left;
    let mut new_y = rect.y - abs_ch_top;
    let mut new_w = rect.width + abs_ch_left + abs_ch_right;
    let mut new_h = rect.height + abs_ch_top + abs_ch_bottom;

    // Clip to container if provided
    if let Some((max_w, max_h)) = container {
        let x0 = new_x.max(0.0);
        let y0 = new_y.max(0.0);
        let x1 = (new_x + new_w).min(max_w);
        let y1 = (new_y + new_h).min(max_h);

        new_x = x0;
        new_y = y0;
        new_w = (x1 - x0).max(0.0);
        new_h = (y1 - y0).max(0.0);
    }

    // Force even dimensions (Only meaningful if pixels)
    if force_even {
        new_x = new_x.round();
        new_y = new_y.round();
        let mut w_int = new_w.round() as i32;
        let mut h_int = new_h.round() as i32;

        if w_int % 2 != 0 { w_int = (w_int - 1).max(0); }
        if h_int % 2 != 0 { h_int = (h_int - 1).max(0); }

        new_w = w_int as f32;
        new_h = h_int as f32;
    }

    Rect { x: new_x, y: new_y, width: new_w, height: new_h }
}

fn normalize_face_rect(rect: Rect, detector: FaceDetector) -> Rect {
    match detector {
        FaceDetector::Default => rect,
        FaceDetector::AppleVision => {
            apply_offsets(rect, VISION_TO_DEFAULT_OFFSETS, None, false)
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), uniffi::export)]
pub fn compute_iou(a: Rect, b: Rect) -> f32 {
    let x_overlap = (a.x + a.width).min(b.x + b.width) - a.x.max(b.x);
    let y_overlap = (a.y + a.height).min(b.y + b.height) - a.y.max(b.y);

    if x_overlap <= 0.0 || y_overlap <= 0.0 {
        return 0.0;
    }

    let intersection = x_overlap * y_overlap;
    let union = (a.width * a.height) + (b.width * b.height) - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

#[cfg_attr(not(target_arch = "wasm32"), uniffi::export)]
pub fn is_contained(inner: Rect, outer: Rect, min_overlap_pct: f32) -> bool {
    let inner_r = inner.x + inner.width;
    let inner_b = inner.y + inner.height;
    
    let outer_r = outer.x + outer.width;
    let outer_b = outer.y + outer.height;

    let visible_l = inner.x.max(outer.x);
    let visible_r = inner_r.min(outer_r);
    let visible_t = inner.y.max(outer.y);
    let visible_b = inner_b.min(outer_b);

    let visible_w = (visible_r - visible_l).max(0.0);
    let visible_h = (visible_b - visible_t).max(0.0);

    let required_w = inner.width * min_overlap_pct;
    let required_h = inner.height * min_overlap_pct;

    visible_w >= required_w && visible_h >= required_h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FaceDetector;

    fn assert_rect_approx_eq(actual: Rect, expected: Rect, epsilon: f32) {
        assert!((actual.x - expected.x).abs() < epsilon, "X mismatch: got {}, expected {}", actual.x, expected.x);
        assert!((actual.y - expected.y).abs() < epsilon, "Y mismatch: got {}, expected {}", actual.y, expected.y);
        assert!((actual.width - expected.width).abs() < epsilon, "W mismatch: got {}, expected {}", actual.width, expected.width);
        assert!((actual.height - expected.height).abs() < epsilon, "H mismatch: got {}, expected {}", actual.height, expected.height);
    }

    #[test]
    fn test_roi_face() {
        let face = Rect::new(100.0, 100.0, 80.0, 120.0);
        let result = calculate_roi(face, RoiMethod::Face, FaceDetector::Default, None, None, false);
        
        assert_rect_approx_eq(result, Rect::new(116.0, 112.0, 48.0, 96.0), 0.1);
    }

    #[test]
    fn test_roi_face_normalized() {        
        let face = Rect::new(0.1, 0.1, 0.08, 0.12);
        let result = calculate_roi(face, RoiMethod::Face, FaceDetector::Default, None, None, false);
        
        assert_rect_approx_eq(result, Rect::new(0.116, 0.112, 0.048, 0.096), 0.0001);
    }

    #[test]
    fn test_roi_forehead() {
        let face = Rect::new(100.0, 100.0, 80.0, 120.0);
        let result = calculate_roi(face, RoiMethod::Forehead, FaceDetector::Default, None, None, false);

        assert_rect_approx_eq(result, Rect::new(128.0, 118.0, 24.0, 12.0), 0.1);
    }

    #[test]
    fn test_roi_forehead_normalized() {        
        let face = Rect::new(0.1, 0.1, 0.08, 0.12);
        let result = calculate_roi(face, RoiMethod::Forehead, FaceDetector::Default, None, None, false);

        assert_rect_approx_eq(result, Rect::new(0.128, 0.118, 0.024, 0.012), 0.0001);
    }

    #[test]
    fn test_roi_upper_body() {
        let face = Rect::new(100.0, 100.0, 80.0, 120.0);
        let result = calculate_roi(face, RoiMethod::UpperBody, FaceDetector::Default, None, None, false);

        assert_rect_approx_eq(result, Rect::new(80.0, 76.0, 120.0, 192.0), 0.1);
    }

    #[test]
    fn test_roi_upper_body_normalized() {        
        let face = Rect::new(0.1, 0.1, 0.08, 0.12);
        let result = calculate_roi(face, RoiMethod::UpperBody, FaceDetector::Default, None, None, false);

        assert_rect_approx_eq(result, Rect::new(0.080, 0.076, 0.120, 0.192), 0.0001);
    }

    #[test]
    fn test_roi_upper_body_cropped() {
        let face = Rect::new(100.0, 100.0, 80.0, 120.0);
        let result = calculate_roi(face, RoiMethod::UpperBodyCropped, FaceDetector::Default, None, None, false);

        assert_rect_approx_eq(result, Rect::new(84.8, 82.54, 110.4, 170.688), 0.01);
    }

    #[test]
    fn test_roi_upper_body_cropped_normalized() {
        let face = Rect::new(0.1, 0.1, 0.08, 0.12);
        let result = calculate_roi(face, RoiMethod::UpperBodyCropped, FaceDetector::Default, None, None, false);

        assert_rect_approx_eq(result, Rect::new(0.0848, 0.08254, 0.1104, 0.170688), 0.0001);
    }

    #[test]
    fn test_clipping_to_container() {
        let face = Rect::new(10.0, 10.0, 50.0, 50.0);
        let container = (40.0, 40.0); 

        let identity = RoiMethod::Custom { left: 0.0, top: 0.0, right: 0.0, bottom: 0.0 };
        let result = calculate_roi(face, identity, FaceDetector::Default, Some(container.0), Some(container.1), false);

        assert_eq!(result.x, 10.0);
        assert_eq!(result.y, 10.0);
        assert_eq!(result.width, 30.0);   
        assert_eq!(result.height, 30.0);  
    }

    #[test]
    fn test_clipping_to_container_normalized() {       
        let face = Rect::new(0.9, 0.9, 0.2, 0.2);
        let container = (1.0, 1.0);
        let identity = RoiMethod::Custom { left: 0.0, top: 0.0, right: 0.0, bottom: 0.0 };

        let result = calculate_roi(face, identity, FaceDetector::Default, Some(container.0), Some(container.1), false);

        assert_eq!(result.x, 0.9);
        assert_eq!(result.y, 0.9);
        assert!((result.width - 0.1).abs() < 0.0001);   
        assert!((result.height - 0.1).abs() < 0.0001);  
    }

    #[test]
    fn test_expansion_clipping() {
        let face = Rect::new(0.0, 0.0, 100.0, 100.0);
        let container = (100.0, 100.0);
        let expand = RoiMethod::Custom { left: 0.5, top: 0.5, right: 0.5, bottom: 0.5 };

        let result = calculate_roi(face, expand, FaceDetector::Default, Some(container.0), Some(container.1), false);

        assert_rect_approx_eq(result, Rect::new(0.0, 0.0, 100.0, 100.0), 0.001);
    }

    #[test]
    fn test_clipping_normalized_negative() {
        let face = Rect::new(-0.1, -0.1, 0.2, 0.2);
        let container = (1.0, 1.0);
        let identity = RoiMethod::Custom { left: 0.0, top: 0.0, right: 0.0, bottom: 0.0 };

        let result = calculate_roi(face, identity, FaceDetector::Default, Some(container.0), Some(container.1), false);

        assert_eq!(result.x, 0.0);
        assert_eq!(result.y, 0.0);
        assert!((result.width - 0.1).abs() < 0.0001);
        assert!((result.height - 0.1).abs() < 0.0001);
    }

    #[test]
    fn test_force_even_dimensions() {
        let face = Rect::new(50.0, 50.0, 51.0, 53.0);
        let identity = RoiMethod::Custom { left: 0.0, top: 0.0, right: 0.0, bottom: 0.0 };

        let result = calculate_roi(face, identity, FaceDetector::Default, None, None, true);

        assert_eq!(result.width, 50.0);
        assert_eq!(result.height, 52.0);
        assert_eq!(result.x, 50.0);
        assert_eq!(result.y, 50.0);
    }

    #[test]
    fn test_iou_calculation() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(5.0, 0.0, 10.0, 10.0);
        
        let iou = compute_iou(a, b);
        assert!((iou - 0.3333).abs() < 0.001);

        let c = Rect::new(20.0, 20.0, 10.0, 10.0);
        assert_eq!(compute_iou(a, c), 0.0);

        assert_eq!(compute_iou(a, a), 1.0);
    }

    #[test]
    fn test_containment() {
        let outer = Rect::new(0.0, 0.0, 100.0, 100.0);
        let inner = Rect::new(10.0, 10.0, 50.0, 50.0);
        let crossing = Rect::new(90.0, 10.0, 50.0, 50.0);  

        assert!(is_contained(inner, outer, 0.99));
        assert!(!is_contained(crossing, outer, 0.5));  
        assert!(is_contained(crossing, outer, 0.1));   
    }
}