#[derive(Debug, Clone)]
pub struct SignalBuffer {
    pub sum: Vec<f32>,
    pub count: Vec<u32>,
    pub unit: Option<String>,
}

impl SignalBuffer {
    pub fn new() -> Self {
        Self {
            sum: Vec::new(),
            count: Vec::new(),
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

        // 1. Handle Overlap
        // We only overlap what we actually have history for.
        let valid_overlap = overlap.min(self.sum.len()).min(new_count);

        if valid_overlap > 0 {
            let start_idx = self.sum.len() - valid_overlap;
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
            for &val in new_slice {
                self.sum.push(if val.is_nan() { 0.0 } else { val });
                self.count.push(1);
            }
        }
    }

    pub fn compute_average(&self) -> Vec<f32> {
        self.sum
            .iter()
            .zip(self.count.iter())
            .map(|(s, c)| if *c > 0 { s / (*c as f32) } else { 0.0 })
            .collect()
    }

    pub fn prune(&mut self, keep_count: usize) {
        if self.sum.len() > keep_count {
            let remove_count = self.sum.len() - keep_count;
            self.sum.drain(0..remove_count);
            self.count.drain(0..remove_count);
        }
    }
}