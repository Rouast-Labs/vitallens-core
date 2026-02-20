use crate::types::{Rect, SessionConfig, BufferConfig, InferenceMode, BufferAction, BufferActionType, InferenceCommand, ExecutionPlan, BufferMetadata};
use crate::geometry::roi;

const MAX_BASE64_BYTES: f64 = 5_760_000.0;
const MAX_STREAM_POLICY_FRAMES: u32 = 150;
const BASE64_OVERHEAD: f64 = 1.3333;

impl BufferConfig {
    pub fn from_session_config(config: &SessionConfig) -> Self {
        let n_inputs = config.n_inputs as u32;
        let input_size = config.input_size as u32;

        let min_with_state = n_inputs;
        let min_no_state = 16.max(n_inputs);

        // Payload capacity calculation
        let bytes_per_frame = (input_size * input_size * 3) as f64;
        let raw_capacity_bytes = MAX_BASE64_BYTES / BASE64_OVERHEAD;
        let calculated_max = (raw_capacity_bytes / bytes_per_frame).floor() as u32;

        let file_max = calculated_max;
        let stream_max = MAX_STREAM_POLICY_FRAMES.min(calculated_max);

        // Standard overlap logic
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

#[derive(Debug)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Object))]
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

    /// Evaluates a target ROI against the list of known buffers.
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

    /// Determines which buffer gets consumed, and which buffers should be dropped.
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

        // Automatically mark any buffers that exceed the timeout limit for deletion
        for meta in &active_buffers {
            if (current_time - meta.last_seen) > self.timeout_seconds {
                buffers_to_drop.push(meta.id.clone());
            }
        }

        // Filter to candidates that are both alive and meet frame count limits
        let mut ready_candidates: Vec<&BufferMetadata> = active_buffers.iter()
            .filter(|meta| !buffers_to_drop.contains(&meta.id))
            .filter(|meta| self.is_ready(meta.count, mode, has_state, flush))
            .collect();

        // Sort descending by creation date (newest buffer wins)
        ready_candidates.sort_by(|a, b| {
            b.created_at.partial_cmp(&a.created_at).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Resolve the winner and generate execution limits
        if let Some(winner) = ready_candidates.first() {
            let winner_id = winner.id.clone();
            let winner_created_at = winner.created_at;

            let take_count = if flush {
                winner.count // TODO: How do we make sure this is not > max?
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

            // Mark any buffer older than the winner for deletion to prevent zombies
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

    /// Determines if a buffer is actionable based on configuration constraints.
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

    /// Helper to easily construct metadata for tests
    fn meta(id: &str, count: u32, created_at: f64, last_seen: f64) -> BufferMetadata {
        BufferMetadata {
            id: id.to_string(),
            roi: Rect::new(0.0, 0.0, 10.0, 10.0),
            count,
            created_at,
            last_seen,
        }
    }

    // ==========================================
    // Configuration Tests
    // ==========================================

    #[test]
    fn test_buffer_config_math() {
        let config = SessionConfig {
            supported_vitals: vec![],
            return_waveforms: None,
            fps_target: 30.0,
            input_size: 100,
            n_inputs: 5,
            roi_method: "face".to_string(),
        };
        let buf_cfg = BufferConfig::from_session_config(&config);
        
        assert_eq!(buf_cfg.min_with_state, 5);
        assert_eq!(buf_cfg.min_no_state, 16); // 16.max(5)
        assert_eq!(buf_cfg.overlap, 4); // 5 - 1
        
        // RAW capacity = MAX_BASE64_BYTES / BASE64_OVERHEAD = 5760000 / 1.3333 ≈ 4320108.0
        // Calculated max frames = 4320108.0 / 30000 = 144
        assert_eq!(buf_cfg.file_max, 144);
        assert_eq!(buf_cfg.stream_max, 144); // 150.min(144)
    }

    // ==========================================
    // Stateless Target Evaluation Tests
    // ==========================================

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
        let roi2 = Rect::new(5.0, 5.0, 100.0, 100.0); // High overlap (IoU)
        
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
        let roi2 = Rect::new(500.0, 500.0, 100.0, 100.0); // No overlap
        
        let active = vec![BufferMetadata { id: "b1".to_string(), roi: roi1, count: 5, created_at: 1.0, last_seen: 1.0 }];

        let res = planner.evaluate_target(roi2, 1.1, active);

        assert_eq!(res.action, BufferActionType::Create);
        assert!(res.matched_id.is_none());
    }

    #[test]
    fn test_evaluate_target_ignores_stale_buffers() {
        let planner = BufferPlanner::new(mock_config());
        let roi1 = Rect::new(0.0, 0.0, 100.0, 100.0);
        
        // This buffer has not been seen in 10 seconds (timeout is 5.0)
        let active = vec![BufferMetadata { id: "b1".to_string(), roi: roi1, count: 5, created_at: 1.0, last_seen: 1.0 }];

        // Evaluate at timestamp 11.0
        let res = planner.evaluate_target(roi1, 11.0, active);
        
        assert_eq!(res.action, BufferActionType::Create, "Should ignore stale buffer and create new");
    }

    // ==========================================
    // Stateless Execution Polling Tests
    // ==========================================

    #[test]
    fn test_file_mode_wait_logic() {
        let planner = BufferPlanner::new(mock_config());

        // Under limit
        let plan1 = planner.poll(vec![meta("b1", 100, 1.0, 1.0)], 1.0, InferenceMode::File, false, false);
        assert!(plan1.command.is_none()); // 100 < file_max (900)

        // Over limit
        let plan2 = planner.poll(vec![meta("b1", 900, 1.0, 1.0)], 1.0, InferenceMode::File, false, false);
        assert!(plan2.command.is_some());
        assert_eq!(plan2.command.unwrap().take_count, 900);
    }

    #[test]
    fn test_file_mode_flush() {
        let planner = BufferPlanner::new(mock_config());
        
        // Force flush override with only 20 frames (limit is 900)
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
        
        // 200 exceeds stream_max (150)
        let plan = planner.poll(vec![meta("b1", 200, 1.0, 1.0)], 1.0, InferenceMode::Stream, false, false);
        let cmd = plan.command.unwrap();
        assert_eq!(cmd.take_count, 150); 
    }

    #[test]
    fn test_file_mode_cap_take_count() {
        let planner = BufferPlanner::new(mock_config());
        
        // 1000 exceeds file_max (900)
        let plan = planner.poll(vec![meta("b1", 1000, 1.0, 1.0)], 1.0, InferenceMode::File, false, false);
        let cmd = plan.command.unwrap();
        assert_eq!(cmd.take_count, 900);
    }

    #[test]
    fn test_stateful_inference_threshold() {
        let planner = BufferPlanner::new(mock_config()); // min_no_state = 16, min_with_state = 4
        
        let active = vec![meta("b1", 5, 1.0, 1.0)];
        
        let plan_no_state = planner.poll(active.clone(), 1.0, InferenceMode::Stream, false, false);
        assert!(plan_no_state.command.is_none()); // 5 < 16, so it waits
        
        let plan_with_state = planner.poll(active, 1.0, InferenceMode::Stream, true, false);
        assert!(plan_with_state.command.is_some()); // 5 >= 4, so it executes
    }

    #[test]
    fn test_overlap_keep_count() {
        let planner = BufferPlanner::new(mock_config()); // overlap = 3
        let plan = planner.poll(vec![meta("b1", 20, 1.0, 1.0)], 1.0, InferenceMode::Stream, false, false);
        let cmd = plan.command.unwrap();
        assert_eq!(cmd.keep_count, 3);
    }

    #[test]
    fn test_poll_multiple_buffers_prioritizes_newest_and_drops_older() {
        let planner = BufferPlanner::new(mock_config());
        
        let active = vec![
            meta("b1", 20, 1.0, 5.0), // Oldest
            meta("b2", 20, 2.0, 5.0), // Middle
            meta("b3", 20, 3.0, 5.0), // Newest
        ];
        
        let plan = planner.poll(active, 5.0, InferenceMode::Stream, false, false);
        
        let cmd = plan.command.unwrap();
        assert_eq!(cmd.buffer_id, "b3"); // Newest wins
        
        // Any ready buffers older than the winner should be dropped
        assert_eq!(plan.buffers_to_drop.len(), 2);
        assert!(plan.buffers_to_drop.contains(&"b1".to_string()));
        assert!(plan.buffers_to_drop.contains(&"b2".to_string()));
    }

    #[test]
    fn test_poll_drops_stale_buffers_even_when_no_winner() {
        let planner = BufferPlanner::new(mock_config());
        
        let active = vec![
            meta("b1", 2, 10.0, 10.0), // Fresh, but not enough frames (2 < 16)
            meta("b2", 2, 1.0, 1.0)    // Stale (Current time 10.0 - 1.0 = 9.0 > 5.0)
        ]; 
        
        let plan = planner.poll(active, 10.0, InferenceMode::Stream, false, false);
        
        assert!(plan.command.is_none()); // Nobody wins
        assert_eq!(plan.buffers_to_drop.len(), 1);
        assert_eq!(plan.buffers_to_drop[0], "b2"); // Stale is culled
    }
}