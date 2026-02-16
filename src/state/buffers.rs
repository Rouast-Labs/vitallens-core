use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct SignalBuffer {
    pub sum: VecDeque<f32>,
    pub count: VecDeque<u32>,
    pub unit: Option<String>,
}

impl SignalBuffer {
    pub fn new() -> Self {
        Self {
            sum: VecDeque::new(),
            count: VecDeque::new(),
            unit: None,
        }
    }

    pub fn merge(&mut self, data: &[f32], overlap: usize, unit: Option<String>) {
        if self.unit.is_none() {
            self.unit = unit;
        }

        let new_count = data.len();
        if new_count == 0 {
            return;
        }

        // 1. Handle Overlap (Add to existing tail)
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

        // 2. Append New Data
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

    /// Computes the average and returns a linear Vec<f32>.
    pub fn compute_average(&self) -> Vec<f32> {
        self.sum
            .iter()
            .zip(self.count.iter())
            .map(|(s, c)| if *c > 0 { s / (*c as f32) } else { 0.0 })
            .collect()
    }

    /// Removes old data from the front.
    pub fn prune(&mut self, keep_count: usize) {
        if self.sum.len() > keep_count {
            let remove_count = self.sum.len() - keep_count;
            self.sum.drain(0..remove_count);
            self.count.drain(0..remove_count);
        }
    }
}

// TODO: Tests