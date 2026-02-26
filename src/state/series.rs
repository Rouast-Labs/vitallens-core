use std::collections::VecDeque;

/// A running buffer designed to stitch together overlapping time-series segments.
/// It maintains a running sum and count for each frame index, allowing it to seamlessly
/// average out overlapping predictions from consecutive inference windows.
#[derive(Debug, Clone)]
pub struct SignalBuffer {
    pub sum: VecDeque<f32>,
    pub count: VecDeque<u32>,
    pub unit: Option<String>,
}

impl SignalBuffer {
    /// Creates a new, empty `SignalBuffer`.
    pub fn new() -> Self {
        Self {
            sum: VecDeque::new(),
            count: VecDeque::new(),
            unit: None,
        }
    }

    /// Merges a new segment of data into the buffer, handling overlapping regions by
    /// accumulating sums and counts. Automatically handles `NaN` values by carrying forward
    /// the last valid value in non-overlapping regions, or ignoring them in overlapping regions.
    ///
    /// # Arguments
    /// * `data` - The new signal segment to append.
    /// * `overlap` - The number of frames at the start of `data` that overlap with the end of the existing buffer.
    /// * `unit` - An optional string representing the unit of measurement.
    pub fn merge(&mut self, data: &[f32], overlap: usize, unit: Option<String>) {
        if self.unit.is_none() {
            self.unit = unit;
        }

        let new_count = data.len();
        if new_count == 0 {
            return;
        }

        let current_len = self.sum.len();
        let valid_overlap = overlap.min(current_len).min(new_count);

        if valid_overlap > 0 {
            let start_idx = current_len - valid_overlap;
            for i in 0..valid_overlap {
                let new_val = data[i];
                if !new_val.is_nan() {
                    self.sum[start_idx + i] += new_val;
                    self.count[start_idx + i] += 1;
                }
            }
        }

        if new_count > valid_overlap {
            let new_slice = &data[valid_overlap..];
            
            let mut last_valid = if let Some(&last) = self.sum.back() {
                 let c = *self.count.back().unwrap_or(&1);
                 if c > 0 { last / c as f32 } else { 0.0 }
            } else {
                0.0 
            };

            for &val in new_slice {
                let val_to_store = if val.is_nan() {
                    last_valid
                } else {
                    last_valid = val;
                    val
                };
                
                self.sum.push_back(val_to_store);
                self.count.push_back(1);
            }
        }
    }

    /// Computes the final continuous signal by dividing the accumulated sums by their respective counts.
    ///
    /// # Returns
    /// A `Vec<f32>` containing the averaged signal.
    pub fn compute_average(&self) -> Vec<f32> {
        self.sum
            .iter()
            .zip(self.count.iter())
            .map(|(s, c)| if *c > 0 { s / (*c as f32) } else { 0.0 })
            .collect()
    }

    /// Trims the oldest elements from the front of the buffer to maintain a maximum history size.
    ///
    /// # Arguments
    /// * `keep_count` - The maximum number of recent frames to retain.
    pub fn prune(&mut self, keep_count: usize) {
        if self.sum.len() > keep_count {
            let remove_count = self.sum.len() - keep_count;
            self.sum.drain(0..remove_count);
            self.count.drain(0..remove_count);
        }
    }

    /// Clears all data and counts from the buffer.
    pub fn clear(&mut self) {
        self.sum.clear();
        self.count.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_appends_data_cleanly() {
        let mut buf = SignalBuffer::new();
        buf.merge(&[1.0, 2.0], 0, Some("bpm".to_string()));
        
        let avg = buf.compute_average();
        assert_eq!(avg, vec![1.0, 2.0]);
        assert_eq!(buf.unit.as_deref(), Some("bpm"));
    }

    #[test]
    fn test_merge_overlap_averages_values() {
        let mut buf = SignalBuffer::new();
        buf.merge(&[10.0, 20.0], 0, None);
        buf.merge(&[30.0, 40.0], 1, None);

        let avg = buf.compute_average();
        assert_eq!(avg, vec![10.0, 25.0, 40.0]);
    }

    #[test]
    fn test_merge_overlap_larger_than_data() {
        let mut buf = SignalBuffer::new();
        buf.merge(&[10.0, 20.0, 30.0], 0, None);
        buf.merge(&[40.0], 1, None);

        let avg = buf.compute_average();
        assert_eq!(avg, vec![10.0, 20.0, 35.0]);
    }

    #[test]
    fn test_nan_in_new_data_is_filled() {
        let mut buf = SignalBuffer::new();
        buf.merge(&[10.0], 0, None);
        buf.merge(&[f32::NAN, 20.0], 0, None);

        let avg = buf.compute_average();
        assert_eq!(avg, vec![10.0, 10.0, 20.0]);
    }

    #[test]
    fn test_initial_nan_becomes_zero() {
        let mut buf = SignalBuffer::new();
        buf.merge(&[f32::NAN, 5.0], 0, None);
        
        let avg = buf.compute_average();
        assert_eq!(avg, vec![0.0, 5.0]);
    }

    #[test]
    fn test_nan_in_overlap_is_ignored() {
        let mut buf = SignalBuffer::new();
        buf.merge(&[10.0, 20.0], 0, None);
        buf.merge(&[f32::NAN, 30.0], 1, None);

        let avg = buf.compute_average();
        assert_eq!(avg, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_pruning_keeps_last_n() {
        let mut buf = SignalBuffer::new();
        buf.merge(&[1.0, 2.0, 3.0, 4.0, 5.0], 0, None);
        
        buf.prune(3);
        
        let avg = buf.compute_average();
        assert_eq!(avg, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_pruning_larger_than_size_does_nothing() {
        let mut buf = SignalBuffer::new();
        buf.merge(&[1.0, 2.0], 0, None);
        
        buf.prune(10);
        
        assert_eq!(buf.compute_average().len(), 2);
    }

    #[test]
    fn test_clear_removes_all_data() {
        let mut buf = SignalBuffer::new();
        buf.merge(&[10.0, 20.0, 30.0], 0, None);
        
        assert_eq!(buf.compute_average().len(), 3);
        
        buf.clear();
        
        assert!(buf.sum.is_empty());
        assert!(buf.count.is_empty());
        assert_eq!(buf.compute_average().len(), 0);
    }
}