use std::collections::HashMap;
use crate::types::{Rect, ModelConfig, BufferConfig, InferenceMode, BufferAction, BufferActionType, InferenceCommand, ExecutionPlan};
use crate::geometry::roi;

const MAX_BASE64_BYTES: f64 = 5_760_000.0;
const MAX_STREAM_POLICY_FRAMES: u32 = 150;
const BASE64_OVERHEAD: f64 = 1.3333;

impl BufferConfig {
    pub fn from_model_config(config: &ModelConfig) -> Self {
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

#[derive(Debug)]
struct GhostBufferMetadata {
    id: String,
    roi: Rect,
    created_at: f64,
    last_seen: f64,
}

#[derive(Debug)]
pub struct BufferManagerCore {
    buffers: HashMap<String, GhostBufferMetadata>,
    config: BufferConfig,
    iou_threshold: f32,
    timeout_seconds: f64,
    next_id: u64,
}

impl BufferManagerCore {
    pub fn new(config: BufferConfig) -> Self {
        Self {
            buffers: HashMap::new(),
            config,
            iou_threshold: 0.6,
            timeout_seconds: 5.0,
            next_id: 0,
        }
    }

    /// Registers a target ROI (from face detection or UI).
    pub fn register_roi(&mut self, target_roi: Rect, timestamp: f64) -> BufferAction {
        self.prune_stale_buffers(timestamp);

        let mut best_id = None;
        let mut max_iou = -1.0;

        for (id, meta) in &self.buffers {
            let iou = roi::compute_iou(target_roi, meta.roi);
            if iou > max_iou {
                max_iou = iou;
                best_id = Some(id.clone());
            }
        }

        if let Some(id) = best_id {
            if max_iou >= self.iou_threshold {
                if let Some(meta) = self.buffers.get_mut(&id) {
                    meta.last_seen = timestamp;
                    return BufferAction { 
                        action: BufferActionType::KeepAlive, 
                        id, 
                        roi: None 
                    };
                }
            }
        }

        let new_id = format!("buf_{}", self.next_id);
        self.next_id += 1;

        let new_meta = GhostBufferMetadata {
            id: new_id.clone(),
            roi: target_roi,
            created_at: timestamp,
            last_seen: timestamp,
        };

        self.buffers.insert(new_id.clone(), new_meta);

        BufferAction { 
            action: BufferActionType::Create, 
            id: new_id, 
            roi: Some(target_roi) 
        }
    }

    /// Polls for a consumption plan based on current native buffer counts.
    pub fn poll(
        &mut self, 
        current_counts: &HashMap<String, u32>, 
        mode: InferenceMode, 
        has_state: bool,
        flush: bool
    ) -> ExecutionPlan {
        
        // 1. Identify Candidates
        let mut ready_candidates: Vec<(&GhostBufferMetadata, u32)> = self.buffers.values()
            .filter_map(|meta| {
                let count = *current_counts.get(&meta.id)?;
                if self.is_ready(count, mode, has_state, flush) {
                    Some((meta, count))
                } else {
                    None
                }
            })
            .collect();

        // 2. Prioritize (Youngest First)
        ready_candidates.sort_by(|(a, _), (b, _)| {
            b.created_at.partial_cmp(&a.created_at).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut command = None;
        let mut buffers_to_drop = Vec::new();

        // 3. Pick Winner
        if let Some((winner_meta, winner_count)) = ready_candidates.first() {
            let winner_id = winner_meta.id.clone();
            let winner_created_at = winner_meta.created_at;

            let take_count = if flush {
                *winner_count
            } else {
                match mode {
                    InferenceMode::Stream => (*winner_count).min(self.config.stream_max),
                    InferenceMode::File => {
                        if *winner_count >= self.config.file_max {
                            self.config.file_max
                        } else {
                            *winner_count
                        }
                    }
                }
            };

            let keep_count = self.config.overlap.min(take_count);

            command = Some(InferenceCommand {
                buffer_id: winner_id,
                take_count,
                keep_count,
            });

            // Kill Logic: Drop all buffers strictly OLDER than the winner.
            for meta in self.buffers.values() {
                if meta.created_at < winner_created_at {
                    buffers_to_drop.push(meta.id.clone());
                }
            }
        }

        // 4. Cleanup Rust State
        for id in &buffers_to_drop {
            self.buffers.remove(id);
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

    pub fn reset(&mut self) {
        self.buffers.clear();
        self.next_id = 0;
    }

    fn prune_stale_buffers(&mut self, current_time: f64) {
        self.buffers.retain(|_, meta| {
            (current_time - meta.last_seen) <= self.timeout_seconds
        });
    }
}

#[derive(Debug)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Object))]
pub struct BufferManager {
    core: std::sync::Mutex<BufferManagerCore>,
}

#[cfg_attr(not(target_arch = "wasm32"), uniffi::export)]
impl BufferManager {
    #[cfg_attr(not(target_arch = "wasm32"), uniffi::constructor)]
    pub fn new(config: BufferConfig) -> Self {
        Self {
            core: std::sync::Mutex::new(BufferManagerCore::new(config)),
        }
    }

    pub fn register_roi(&self, target_roi: Rect, timestamp: f64) -> BufferAction {
        self.core.lock().unwrap().register_roi(target_roi, timestamp)
    }

    pub fn poll(
        &self, 
        current_counts: HashMap<String, u32>, 
        mode: InferenceMode, 
        has_state: bool, 
        flush: bool
    ) -> ExecutionPlan {
        self.core.lock().unwrap().poll(&current_counts, mode, has_state, flush)
    }

    pub fn reset(&self) {
        self.core.lock().unwrap().reset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelConfig;

    fn mock_config() -> BufferConfig {
        BufferConfig {
            min_no_state: 16,
            min_with_state: 4,
            stream_max: 150,
            file_max: 900,
            overlap: 3,  
        }
    }

    // --- Configuration Tests ---

    #[test]
    fn test_buffer_config_math() {
        let config = ModelConfig {
            name: "test".to_string(),
            supported_vitals: vec![],
            fps_target: 30.0,
            input_size: 100, // bytes_per_frame = 100 * 100 * 3 = 30_000
            n_inputs: 5,
            roi_method: "face".to_string(),
        };
        let buf_cfg = BufferConfig::from_model_config(&config);
        
        assert_eq!(buf_cfg.min_with_state, 5);
        assert_eq!(buf_cfg.min_no_state, 16); // 16.max(5)
        assert_eq!(buf_cfg.overlap, 4); // 5 - 1
        
        // RAW capacity = MAX_BASE64_BYTES / BASE64_OVERHEAD = 5760000 / 1.3333 ≈ 4320108.0
        // Calculated max frames = 4320108.0 / 30000 = 144
        assert_eq!(buf_cfg.file_max, 144);
        assert_eq!(buf_cfg.stream_max, 144); // 150.min(144)
    }

    // --- ROI Registration Tests ---

    #[test]
    fn test_register_roi_creates_new() {
        let mut mgr = BufferManagerCore::new(mock_config());
        let roi = Rect::new(0.0, 0.0, 100.0, 100.0);
        
        let res = mgr.register_roi(roi, 1.0);
        
        assert_eq!(res.action, BufferActionType::Create);
        assert_eq!(res.id, "buf_0");
        assert!(res.roi.is_some());
    }

    #[test]
    fn test_register_roi_keeps_alive() {
        let mut mgr = BufferManagerCore::new(mock_config());
        let roi1 = Rect::new(0.0, 0.0, 100.0, 100.0);
        let roi2 = Rect::new(5.0, 5.0, 100.0, 100.0); // High overlap (IoU)

        mgr.register_roi(roi1, 1.0); 
        let res = mgr.register_roi(roi2, 1.1); 

        assert_eq!(res.action, BufferActionType::KeepAlive);
        assert_eq!(res.id, "buf_0");
        assert!(res.roi.is_none());
    }

    #[test]
    fn test_register_roi_low_iou_creates_new() {
        let mut mgr = BufferManagerCore::new(mock_config());
        let roi1 = Rect::new(0.0, 0.0, 100.0, 100.0);
        let roi2 = Rect::new(500.0, 500.0, 100.0, 100.0); // No overlap

        let res1 = mgr.register_roi(roi1, 1.0);
        let res2 = mgr.register_roi(roi2, 2.0);

        assert_eq!(res1.action, BufferActionType::Create);
        assert_eq!(res2.action, BufferActionType::Create);
        assert_ne!(res1.id, res2.id); // Should be buf_0 and buf_1
    }

    #[test]
    fn test_register_roi_prunes_stale() {
        let mut mgr = BufferManagerCore::new(mock_config());
        let roi = Rect::new(0.0, 0.0, 100.0, 100.0);
        
        mgr.register_roi(roi, 1.0); // Creates buf_0, last_seen = 1.0
        
        // timeout_seconds is 5.0. 
        // Polling at 7.0 means (7.0 - 1.0) = 6.0 > 5.0, so buf_0 is stale and gets pruned.
        let res = mgr.register_roi(roi, 7.0); 
        
        assert_eq!(res.action, BufferActionType::Create);
        assert_eq!(res.id, "buf_1"); // Must be a new ID since buf_0 was dropped
    }

    // --- Execution Polling & Threshold Tests ---

    #[test]
    fn test_file_mode_wait_logic() {
        let mut mgr = BufferManagerCore::new(mock_config());
        mgr.register_roi(Rect::new(0.0, 0.0, 100.0, 100.0), 1.0); 

        let mut counts = HashMap::new();
        
        counts.insert("buf_0".to_string(), 100);
        let plan = mgr.poll(&counts, InferenceMode::File, false, false);
        assert!(plan.command.is_none()); // 100 < file_max (900)

        counts.insert("buf_0".to_string(), 900);
        let plan = mgr.poll(&counts, InferenceMode::File, false, false);
        assert!(plan.command.is_some());
        assert_eq!(plan.command.unwrap().take_count, 900);
    }

    #[test]
    fn test_file_mode_flush() {
        let mut mgr = BufferManagerCore::new(mock_config());
        mgr.register_roi(Rect::new(0.0, 0.0, 100.0, 100.0), 1.0); 

        let mut counts = HashMap::new();
        counts.insert("buf_0".to_string(), 20);
        
        // Force flush override
        let plan = mgr.poll(&counts, InferenceMode::File, false, true);
        
        assert!(plan.command.is_some());
        assert_eq!(plan.command.unwrap().take_count, 20);
    }

    #[test]
    fn test_stream_mode_latency() {
        let mut mgr = BufferManagerCore::new(mock_config());
        mgr.register_roi(Rect::new(0.0, 0.0, 100.0, 100.0), 1.0); 

        let mut counts = HashMap::new();
        counts.insert("buf_0".to_string(), 16);
        
        let plan = mgr.poll(&counts, InferenceMode::Stream, false, false);
        assert!(plan.command.is_some());
    }

    #[test]
    fn test_stream_mode_cap_take_count() {
        let mut mgr = BufferManagerCore::new(mock_config());
        mgr.register_roi(Rect::new(0.0, 0.0, 10.0, 10.0), 1.0);
        
        let mut counts = HashMap::new();
        counts.insert("buf_0".to_string(), 200); // Exceeds stream_max (150)
        
        let plan = mgr.poll(&counts, InferenceMode::Stream, false, false);
        let cmd = plan.command.unwrap();
        assert_eq!(cmd.take_count, 150); 
    }

    #[test]
    fn test_file_mode_cap_take_count() {
        let mut mgr = BufferManagerCore::new(mock_config());
        mgr.register_roi(Rect::new(0.0, 0.0, 10.0, 10.0), 1.0);
        
        let mut counts = HashMap::new();
        counts.insert("buf_0".to_string(), 1000); // Exceeds file_max (900)
        
        let plan = mgr.poll(&counts, InferenceMode::File, false, false);
        let cmd = plan.command.unwrap();
        assert_eq!(cmd.take_count, 900);
    }

    #[test]
    fn test_stateful_inference_threshold() {
        let mut mgr = BufferManagerCore::new(mock_config());
        mgr.register_roi(Rect::new(0.0, 0.0, 10.0, 10.0), 1.0);
        
        let mut counts = HashMap::new();
        counts.insert("buf_0".to_string(), 5); // min_no_state = 16, min_with_state = 4
        
        let plan_no_state = mgr.poll(&counts, InferenceMode::Stream, false, false);
        assert!(plan_no_state.command.is_none()); // 5 < 16, so it waits
        
        let plan_with_state = mgr.poll(&counts, InferenceMode::Stream, true, false);
        assert!(plan_with_state.command.is_some()); // 5 >= 4, so it executes
    }

    #[test]
    fn test_overlap_keep_count() {
        let mut mgr = BufferManagerCore::new(mock_config()); // overlap = 3
        mgr.register_roi(Rect::new(0.0, 0.0, 10.0, 10.0), 1.0);
        
        let mut counts = HashMap::new();
        counts.insert("buf_0".to_string(), 20);
        
        let plan = mgr.poll(&counts, InferenceMode::Stream, false, false);
        let cmd = plan.command.unwrap();
        assert_eq!(cmd.keep_count, 3);
    }

    #[test]
    fn test_poll_multiple_buffers_prioritizes_newest_and_drops_older() {
        let mut mgr = BufferManagerCore::new(mock_config());
        
        // Register multiple buffers over time
        mgr.register_roi(Rect::new(0.0, 0.0, 10.0, 10.0), 1.0);     // buf_0 (Oldest)
        mgr.register_roi(Rect::new(500.0, 500.0, 10.0, 10.0), 2.0); // buf_1
        mgr.register_roi(Rect::new(1000.0, 1000.0, 10.0, 10.0), 3.0); // buf_2 (Newest)
        
        let mut counts = HashMap::new();
        counts.insert("buf_0".to_string(), 20); 
        counts.insert("buf_1".to_string(), 20); 
        counts.insert("buf_2".to_string(), 20); 
        
        // The Core sorts b.created_at.cmp(a.created_at), so newest wins.
        let plan = mgr.poll(&counts, InferenceMode::Stream, false, false);
        
        let cmd = plan.command.unwrap();
        assert_eq!(cmd.buffer_id, "buf_2"); // Newest should be prioritized
        
        // Since buf_2 is the winner, any ready buffers older than it (0 and 1) 
        // should be moved to the drop list to avoid processing stale data.
        assert_eq!(plan.buffers_to_drop.len(), 2);
        assert!(plan.buffers_to_drop.contains(&"buf_0".to_string()));
        assert!(plan.buffers_to_drop.contains(&"buf_1".to_string()));
    }

    // --- State Management ---

    #[test]
    fn test_reset_clears_state() {
        let mut mgr = BufferManagerCore::new(mock_config());
        mgr.register_roi(Rect::new(0.0, 0.0, 10.0, 10.0), 1.0); // creates buf_0
        
        mgr.reset();
        
        let res = mgr.register_roi(Rect::new(0.0, 0.0, 10.0, 10.0), 2.0); 
        assert_eq!(res.action, BufferActionType::Create);
        assert_eq!(res.id, "buf_0"); // Counter should be back to 0
    }
}