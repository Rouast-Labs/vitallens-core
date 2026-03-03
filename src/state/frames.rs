use crate::types::{Rect, SessionConfig, BufferConfig, InferenceMode, BufferAction, BufferActionType, InferenceCommand, ExecutionPlan, BufferMetadata};
use crate::geometry::roi;

const MAX_BASE64_BYTES: f64 = 5_760_000.0;
const MAX_STREAM_POLICY_FRAMES: u32 = 150;
const BASE64_OVERHEAD: f64 = 1.3333;

impl BufferConfig {
    /// Generates a buffer configuration dynamically based on the session parameters 
    /// and network payload constraints.
    pub fn from_session_config(config: &SessionConfig) -> Self {
        let n_inputs = config.n_inputs as u32;
        let input_size = config.input_size as u32;

        let min_with_state = n_inputs;
        let min_no_state = 16.max(n_inputs);

        let bytes_per_frame = (input_size * input_size * 3) as f64;
        let raw_capacity_bytes = MAX_BASE64_BYTES / BASE64_OVERHEAD;
        let calculated_max = (raw_capacity_bytes / bytes_per_frame).floor() as u32;

        let file_max = calculated_max;
        let stream_max = MAX_STREAM_POLICY_FRAMES.min(calculated_max);

        let overlap = n_inputs.saturating_sub(1);

        Self {
            min_no_state,
            min_with_state,
            stream_max,
            file_max,
            overlap,
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), uniffi::export)]
pub fn compute_buffer_config(config: SessionConfig) -> BufferConfig {
    BufferConfig::from_session_config(&config)
}

/// Manages the tracking, matching, and lifecycle of video frame buffers across a session.
#[derive(Debug)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Object))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct BufferPlanner {
    config: BufferConfig,
    iou_threshold: f32,
    timeout_seconds: f64,
}

#[cfg_attr(not(target_arch = "wasm32"), uniffi::export)]
impl BufferPlanner {
    #[cfg_attr(not(target_arch = "wasm32"), uniffi::constructor)]
    pub fn new(config: BufferConfig) -> Self {
        Self {
            config,
            iou_threshold: 0.6,
            timeout_seconds: 5.0,
        }
    }

    /// Evaluates a target ROI against the list of known active buffers using 
    /// Intersection over Union (IoU) to maintain spatial consistency.
    ///
    /// # Arguments
    /// * `target_roi` - The bounding box of the newly detected subject.
    /// * `timestamp` - The current frame's timestamp.
    /// * `active_buffers` - Metadata of currently tracked buffers.
    ///
    /// # Returns
    /// A `BufferAction` instructing the caller to either append to an existing buffer or create a new one.
    pub fn evaluate_target(&self, target_roi: Rect, timestamp: f64, active_buffers: Vec<BufferMetadata>) -> BufferAction {
        let mut best_id = None;
        let mut max_iou = -1.0;

        for meta in &active_buffers {
            if (timestamp - meta.last_seen) > self.timeout_seconds {
                continue;
            }

            let iou = roi::compute_iou(target_roi, meta.roi);
            if iou > max_iou {
                max_iou = iou;
                best_id = Some(meta.id.clone());
            }
        }

        if let Some(id) = best_id {
            if max_iou >= self.iou_threshold {
                return BufferAction { 
                    action: BufferActionType::KeepAlive, 
                    matched_id: Some(id), 
                    roi: None 
                };
            }
        }

        BufferAction { 
            action: BufferActionType::Create, 
            matched_id: None, 
            roi: Some(target_roi) 
        }
    }

    /// Determines which buffer (if any) is ready for inference execution, and identifies 
    /// stale buffers that should be dropped from memory.
    ///
    /// # Arguments
    /// * `active_buffers` - Metadata of currently tracked buffers.
    /// * `current_time` - The current execution timestamp.
    /// * `mode` - The operational mode (Stream or File).
    /// * `has_state` - Whether the engine has warmed up with prior state.
    /// * `flush` - If true, forces the consumption of the buffer regardless of capacity limits.
    ///
    /// # Returns
    /// An `ExecutionPlan` containing an optional command to execute inference, and a list of buffer IDs to drop.
    pub fn poll(
        &self, 
        active_buffers: Vec<BufferMetadata>, 
        current_time: f64,
        mode: InferenceMode, 
        has_state: bool, 
        flush: bool
    ) -> ExecutionPlan {
        
        let mut command = None;
        let mut buffers_to_drop = Vec::new();

        for meta in &active_buffers {
            if (current_time - meta.last_seen) > self.timeout_seconds {
                buffers_to_drop.push(meta.id.clone());
            }
        }

        let mut ready_candidates: Vec<&BufferMetadata> = active_buffers.iter()
            .filter(|meta| !buffers_to_drop.contains(&meta.id))
            .filter(|meta| self.is_ready(meta.count, mode, has_state, flush))
            .collect();

        ready_candidates.sort_by(|a, b| {
            b.created_at.partial_cmp(&a.created_at).unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(winner) = ready_candidates.first() {
            let winner_id = winner.id.clone();
            let winner_created_at = winner.created_at;

            let take_count = if flush {
                winner.count.min(self.config.file_max)
            } else {
                match mode {
                    InferenceMode::Stream => winner.count.min(self.config.stream_max),
                    InferenceMode::File => winner.count.min(self.config.file_max),
                }
            };

            let keep_count = self.config.overlap.min(take_count);

            command = Some(InferenceCommand {
                buffer_id: winner_id,
                take_count,
                keep_count,
            });

            for meta in &active_buffers {
                if meta.created_at < winner_created_at && !buffers_to_drop.contains(&meta.id) {
                    buffers_to_drop.push(meta.id.clone());
                }
            }
        }

        ExecutionPlan {
            command,
            buffers_to_drop,
        }
    }

    /// Determines if a specific buffer is actionable based on volume and configuration constraints.
    fn is_ready(&self, count: u32, mode: InferenceMode, has_state: bool, flush: bool) -> bool {
        let min_required = if has_state { self.config.min_with_state } else { self.config.min_no_state };

        if count < min_required { return false; }
        if flush { return true; }

        match mode {
            InferenceMode::Stream => true,
            InferenceMode::File => count >= self.config.file_max,
        }
    }
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl BufferPlanner {
    #[wasm_bindgen(constructor)]
    pub fn new_js(config_val: JsValue) -> Result<BufferPlanner, JsError> {
        let config: crate::types::BufferConfig = serde_wasm_bindgen::from_value(config_val)?;
        Ok(Self::new(config))
    }

    #[wasm_bindgen(js_name = evaluateTarget)]
    pub fn evaluate_target_js(&self, target_roi_val: JsValue, timestamp: f64, active_buffers_val: JsValue) -> Result<JsValue, JsError> {
        let target_roi = serde_wasm_bindgen::from_value(target_roi_val)?;
        let active_buffers = serde_wasm_bindgen::from_value(active_buffers_val)?;
        let action = self.evaluate_target(target_roi, timestamp, active_buffers);
        Ok(serde_wasm_bindgen::to_value(&action)?)
    }

    #[wasm_bindgen(js_name = poll)]
    pub fn poll_js(&self, active_buffers_val: JsValue, current_time: f64, mode_val: JsValue, has_state: bool, flush: bool) -> Result<JsValue, JsError> {
        let active_buffers = serde_wasm_bindgen::from_value(active_buffers_val)?;
        let mode = serde_wasm_bindgen::from_value(mode_val)?;
        let plan = self.poll(active_buffers, current_time, mode, has_state, flush);
        Ok(serde_wasm_bindgen::to_value(&plan)?)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = computeBufferConfig)]
pub fn compute_buffer_config_js(config_val: JsValue) -> Result<JsValue, JsError> {
    let config: crate::types::SessionConfig = serde_wasm_bindgen::from_value(config_val)?;
    let buffer_config = compute_buffer_config(config);
    Ok(serde_wasm_bindgen::to_value(&buffer_config)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SessionConfig;

    fn mock_config() -> BufferConfig {
        BufferConfig {
            min_no_state: 16,
            min_with_state: 4,
            stream_max: 150,
            file_max: 900,
            overlap: 3,  
        }
    }

    fn meta(id: &str, count: u32, created_at: f64, last_seen: f64) -> BufferMetadata {
        BufferMetadata {
            id: id.to_string(),
            roi: Rect::new(0.0, 0.0, 10.0, 10.0),
            count,
            created_at,
            last_seen,
        }
    }

    #[test]
    fn test_buffer_config_math() {
        let config = SessionConfig {
            model_name: "vitallens-2.0".to_string(),
            supported_vitals: vec![],
            return_waveforms: None,
            fps_target: 30.0,
            input_size: 100,
            n_inputs: 5,
            roi_method: "face".to_string(),
        };
        let buf_cfg = BufferConfig::from_session_config(&config);
        
        assert_eq!(buf_cfg.min_with_state, 5);
        assert_eq!(buf_cfg.min_no_state, 16);  
        assert_eq!(buf_cfg.overlap, 4);  
        
        assert_eq!(buf_cfg.file_max, 144);
        assert_eq!(buf_cfg.stream_max, 144);  
    }

    #[test]
    fn test_evaluate_target_creates_new_when_empty() {
        let planner = BufferPlanner::new(mock_config());
        let roi = Rect::new(0.0, 0.0, 100.0, 100.0);
        
        let res = planner.evaluate_target(roi, 1.0, vec![]);
        
        assert_eq!(res.action, BufferActionType::Create);
        assert!(res.matched_id.is_none());
        assert!(res.roi.is_some());
    }

    #[test]
    fn test_evaluate_target_keeps_alive_on_match() {
        let planner = BufferPlanner::new(mock_config());
        let roi1 = Rect::new(0.0, 0.0, 100.0, 100.0);
        let roi2 = Rect::new(5.0, 5.0, 100.0, 100.0); 
        
        let active = vec![BufferMetadata { id: "b1".to_string(), roi: roi1, count: 5, created_at: 1.0, last_seen: 1.0 }];

        let res = planner.evaluate_target(roi2, 1.1, active);

        assert_eq!(res.action, BufferActionType::KeepAlive);
        assert_eq!(res.matched_id.unwrap(), "b1");
        assert!(res.roi.is_none());
    }

    #[test]
    fn test_evaluate_target_low_iou_creates_new() {
        let planner = BufferPlanner::new(mock_config());
        let roi1 = Rect::new(0.0, 0.0, 100.0, 100.0);
        let roi2 = Rect::new(500.0, 500.0, 100.0, 100.0); 
        
        let active = vec![BufferMetadata { id: "b1".to_string(), roi: roi1, count: 5, created_at: 1.0, last_seen: 1.0 }];

        let res = planner.evaluate_target(roi2, 1.1, active);

        assert_eq!(res.action, BufferActionType::Create);
        assert!(res.matched_id.is_none());
    }

    #[test]
    fn test_evaluate_target_ignores_stale_buffers() {
        let planner = BufferPlanner::new(mock_config());
        let roi1 = Rect::new(0.0, 0.0, 100.0, 100.0);
        
        let active = vec![BufferMetadata { id: "b1".to_string(), roi: roi1, count: 5, created_at: 1.0, last_seen: 1.0 }];

        let res = planner.evaluate_target(roi1, 11.0, active);
        
        assert_eq!(res.action, BufferActionType::Create);
    }

    #[test]
    fn test_file_mode_wait_logic() {
        let planner = BufferPlanner::new(mock_config());

        let plan1 = planner.poll(vec![meta("b1", 100, 1.0, 1.0)], 1.0, InferenceMode::File, false, false);
        assert!(plan1.command.is_none());  

        let plan2 = planner.poll(vec![meta("b1", 900, 1.0, 1.0)], 1.0, InferenceMode::File, false, false);
        assert!(plan2.command.is_some());
        assert_eq!(plan2.command.unwrap().take_count, 900);
    }

    #[test]
    fn test_file_mode_flush() {
        let planner = BufferPlanner::new(mock_config());
        
        let plan = planner.poll(vec![meta("b1", 20, 1.0, 1.0)], 1.0, InferenceMode::File, false, true);
        
        assert!(plan.command.is_some());
        assert_eq!(plan.command.unwrap().take_count, 20);
    }

    #[test]
    fn test_stream_mode_latency() {
        let planner = BufferPlanner::new(mock_config());
        let plan = planner.poll(vec![meta("b1", 16, 1.0, 1.0)], 1.0, InferenceMode::Stream, false, false);
        assert!(plan.command.is_some());
    }

    #[test]
    fn test_stream_mode_cap_take_count() {
        let planner = BufferPlanner::new(mock_config());
        
        let plan = planner.poll(vec![meta("b1", 200, 1.0, 1.0)], 1.0, InferenceMode::Stream, false, false);
        let cmd = plan.command.unwrap();
        assert_eq!(cmd.take_count, 150); 
    }

    #[test]
    fn test_file_mode_cap_take_count() {
        let planner = BufferPlanner::new(mock_config());
        
        let plan = planner.poll(vec![meta("b1", 1000, 1.0, 1.0)], 1.0, InferenceMode::File, false, false);
        let cmd = plan.command.unwrap();
        assert_eq!(cmd.take_count, 900);
    }

    #[test]
    fn test_stateful_inference_threshold() {
        let planner = BufferPlanner::new(mock_config());  
        
        let active = vec![meta("b1", 5, 1.0, 1.0)];
        
        let plan_no_state = planner.poll(active.clone(), 1.0, InferenceMode::Stream, false, false);
        assert!(plan_no_state.command.is_none());  
        
        let plan_with_state = planner.poll(active, 1.0, InferenceMode::Stream, true, false);
        assert!(plan_with_state.command.is_some());  
    }

    #[test]
    fn test_overlap_keep_count() {
        let planner = BufferPlanner::new(mock_config());  
        let plan = planner.poll(vec![meta("b1", 20, 1.0, 1.0)], 1.0, InferenceMode::Stream, false, false);
        let cmd = plan.command.unwrap();
        assert_eq!(cmd.keep_count, 3);
    }

    #[test]
    fn test_poll_multiple_buffers_prioritizes_newest_and_drops_older() {
        let planner = BufferPlanner::new(mock_config());
        
        let active = vec![
            meta("b1", 20, 1.0, 5.0),  
            meta("b2", 20, 2.0, 5.0),  
            meta("b3", 20, 3.0, 5.0),  
        ];
        
        let plan = planner.poll(active, 5.0, InferenceMode::Stream, false, false);
        
        let cmd = plan.command.unwrap();
        assert_eq!(cmd.buffer_id, "b3");  
        
        assert_eq!(plan.buffers_to_drop.len(), 2);
        assert!(plan.buffers_to_drop.contains(&"b1".to_string()));
        assert!(plan.buffers_to_drop.contains(&"b2".to_string()));
    }

    #[test]
    fn test_poll_drops_stale_buffers_even_when_no_winner() {
        let planner = BufferPlanner::new(mock_config());
        
        let active = vec![
            meta("b1", 2, 10.0, 10.0),  
            meta("b2", 2, 1.0, 1.0)     
        ]; 
        
        let plan = planner.poll(active, 10.0, InferenceMode::Stream, false, false);
        
        assert!(plan.command.is_none());  
        assert_eq!(plan.buffers_to_drop.len(), 1);
        assert_eq!(plan.buffers_to_drop[0], "b2");  
    }
}