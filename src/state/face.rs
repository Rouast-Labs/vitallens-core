use std::sync::Mutex;
use crate::types::Rect;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// How long the last detected face is kept while detections find no face, in seconds.
pub const DEFAULT_FACE_HOLD_SECONDS: f64 = 1.5;

#[derive(Debug)]
struct FaceHoldCore {
    hold_seconds: f64,
    face: Option<Rect>,
    last_seen: f64,
}

impl FaceHoldCore {
    fn update(&mut self, face: Option<Rect>, timestamp: f64) -> Option<Rect> {
        if timestamp < self.last_seen {
            self.face = None;
        }
        match face {
            Some(rect) => {
                self.face = Some(rect);
                self.last_seen = timestamp;
            }
            None => {
                if self.face.is_some() && timestamp - self.last_seen > self.hold_seconds {
                    self.face = None;
                }
            }
        }
        self.face
    }
}

/// Keeps the last detected face through short gaps in face detection.
///
/// Live face detection misses a face now and then, for example on a blink, a turn of the head,
/// or a frame the detector doesn't handle. `FaceHold` returns the last detected face until
/// detections have found no face for longer than the hold time, and only then reports no face.
#[derive(Debug)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Object))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct FaceHold {
    core: Mutex<FaceHoldCore>,
}

#[cfg_attr(not(target_arch = "wasm32"), uniffi::export)]
impl FaceHold {
    /// Creates a face hold. Without `hold_seconds`, it uses `DEFAULT_FACE_HOLD_SECONDS`.
    #[cfg_attr(not(target_arch = "wasm32"), uniffi::constructor)]
    pub fn new(hold_seconds: Option<f64>) -> Self {
        Self {
            core: Mutex::new(FaceHoldCore {
                hold_seconds: hold_seconds.unwrap_or(DEFAULT_FACE_HOLD_SECONDS),
                face: None,
                last_seen: f64::NEG_INFINITY,
            }),
        }
    }

    /// Registers the result of one face detection and returns the face to use.
    ///
    /// # Arguments
    /// * `face` - The detected face, or `None` if the detection found no face.
    /// * `timestamp` - The time of the detected frame, in seconds. A time earlier than the last
    ///   detected face starts afresh.
    ///
    /// # Returns
    /// The detected face, the last detected face if it was seen within the hold time, or `None`.
    pub fn update(&self, face: Option<Rect>, timestamp: f64) -> Option<Rect> {
        self.core.lock().unwrap().update(face, timestamp)
    }

    /// Forgets the last detected face.
    pub fn reset(&self) {
        let mut core = self.core.lock().unwrap();
        core.face = None;
        core.last_seen = f64::NEG_INFINITY;
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl FaceHold {
    #[new]
    #[pyo3(signature = (hold_seconds=None))]
    pub fn py_new(hold_seconds: Option<f64>) -> Self {
        Self::new(hold_seconds)
    }

    #[pyo3(name = "update", signature = (face, timestamp))]
    pub fn py_update(&self, face: Option<Rect>, timestamp: f64) -> Option<Rect> {
        self.update(face, timestamp)
    }

    #[pyo3(name = "reset")]
    pub fn py_reset(&self) {
        self.reset()
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl FaceHold {
    #[wasm_bindgen(constructor)]
    pub fn new_js(hold_seconds: Option<f64>) -> FaceHold {
        Self::new(hold_seconds)
    }

    #[wasm_bindgen(js_name = update)]
    pub fn update_js(&self, face_val: JsValue, timestamp: f64) -> Result<JsValue, JsError> {
        let face: Option<Rect> = serde_wasm_bindgen::from_value(face_val)?;
        Ok(serde_wasm_bindgen::to_value(&self.update(face, timestamp))?)
    }

    #[wasm_bindgen(js_name = reset)]
    pub fn reset_js(&self) {
        self.reset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f32) -> Rect {
        Rect { x, y: 10.0, width: 100.0, height: 100.0 }
    }

    #[test]
    fn test_no_face_before_first_detection() {
        let hold = FaceHold::new(None);
        assert_eq!(hold.update(None, 0.0), None);
    }

    #[test]
    fn test_returns_detected_face() {
        let hold = FaceHold::new(None);
        assert_eq!(hold.update(Some(rect(1.0)), 0.0), Some(rect(1.0)));
        assert_eq!(hold.update(Some(rect(2.0)), 0.5), Some(rect(2.0)));
    }

    #[test]
    fn test_keeps_face_through_short_gap() {
        let hold = FaceHold::new(None);
        hold.update(Some(rect(1.0)), 0.0);
        assert_eq!(hold.update(None, 0.5), Some(rect(1.0)));
        assert_eq!(hold.update(None, 1.0), Some(rect(1.0)));
        assert_eq!(hold.update(None, 1.5), Some(rect(1.0)));
        assert_eq!(hold.update(Some(rect(2.0)), 2.0), Some(rect(2.0)));
        assert_eq!(hold.update(None, 3.0), Some(rect(2.0)));
    }

    #[test]
    fn test_drops_face_after_hold_time() {
        let hold = FaceHold::new(None);
        hold.update(Some(rect(1.0)), 0.0);
        assert_eq!(hold.update(None, 1.6), None);
        assert_eq!(hold.update(None, 1.7), None);
        assert_eq!(hold.update(Some(rect(2.0)), 2.0), Some(rect(2.0)));
    }

    #[test]
    fn test_custom_hold_time() {
        let hold = FaceHold::new(Some(0.0));
        hold.update(Some(rect(1.0)), 0.0);
        assert_eq!(hold.update(None, 0.0), Some(rect(1.0)));
        assert_eq!(hold.update(None, 0.5), None);
    }

    #[test]
    fn test_earlier_timestamp_starts_afresh() {
        let hold = FaceHold::new(None);
        hold.update(Some(rect(1.0)), 10.0);
        assert_eq!(hold.update(None, 5.0), None);
        assert_eq!(hold.update(Some(rect(2.0)), 5.5), Some(rect(2.0)));
    }

    #[test]
    fn test_reset() {
        let hold = FaceHold::new(None);
        hold.update(Some(rect(1.0)), 0.0);
        hold.reset();
        assert_eq!(hold.update(None, 0.5), None);
    }
}
