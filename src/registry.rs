#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VitalType {
    Provided,
    Derived,
}

#[derive(Debug, Clone)]
pub struct DerivationConfig {
    pub source_signal: String,
    pub method: CalculationMethod,
    pub min_required_seconds: f32, // TODO: Rename
    pub optimal_window_seconds: f32, // TODO: Rename
    pub min_value: f32,
    pub max_value: f32,
    pub order: u8, 
}

#[derive(Debug, Clone, Copy)]
pub enum CalculationMethod {
    RateFromFFT,
    HrvFromPeaks(HrvMetric),
    Average,
}

#[derive(Debug, Clone, Copy)]
pub enum HrvMetric {
    Sdnn,
    Rmssd,
    LfHf,
}

#[derive(Debug, Clone, Copy)]
pub enum PostProcessOp {
    None,
    Detrend,
    Standardize,
}

#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    pub operation: PostProcessOp, 
    pub min_window_seconds: f32,
}

#[derive(Debug, Clone)]
pub struct VitalMeta {
    pub id: String,
    pub vital_type: VitalType,
    pub derivation: Option<DerivationConfig>,
    pub processing: Option<ProcessingConfig>,
    pub unit: String,
    pub display_name: String,
}

// --- THE REGISTRY ---

pub fn get_vital_meta(vital_id: &str) -> Option<VitalMeta> {
    match vital_id {
        "ppg_waveform" | "ppg" => Some(VitalMeta {
            id: "ppg_waveform".to_string(),
            vital_type: VitalType::Provided,
            derivation: None,
            processing: Some(ProcessingConfig {
                operation: PostProcessOp::Detrend,
                min_window_seconds: 1.0,
            }),
            unit: "unitless".to_string(),
            display_name: "PPG Waveform".to_string(),
        }),
        "heart_rate" | "hr" | "pulse" => Some(VitalMeta {
            id: "heart_rate".to_string(),
            vital_type: VitalType::Derived,
            derivation: Some(DerivationConfig {
                source_signal: "ppg_waveform".to_string(),
                method: CalculationMethod::RateFromFFT,
                min_required_seconds: 4.0,
                optimal_window_seconds: 10.0,
                min_value: 40.0,
                max_value: 240.0,
                order: 0,
            }),
            processing: None,
            unit: "bpm".to_string(),
            display_name: "Heart Rate".to_string(),
        }),
        "hrv_sdnn" | "sdnn" => Some(VitalMeta {
            id: "hrv_sdnn".to_string(),
            vital_type: VitalType::Derived,
            derivation: Some(DerivationConfig {
                source_signal: "ppg_waveform".to_string(),
                method: CalculationMethod::HrvFromPeaks(HrvMetric::Sdnn),
                min_required_seconds: 10.0,
                optimal_window_seconds: 60.0,
                min_value: 0.0,
                max_value: 500.0,
                order: 1, 
            }),
            processing: None,
            unit: "ms".to_string(),
            display_name: "HRV (SDNN)".to_string(),
        }),
        "spo2" => Some(VitalMeta {
            id: "spo2".to_string(),
            vital_type: VitalType::Provided,
            derivation: Some(DerivationConfig {
                source_signal: "spo2".to_string(),
                method: CalculationMethod::Average,
                min_required_seconds: 1.0,
                optimal_window_seconds: 5.0,
                min_value: 70.0,
                max_value: 100.0,
                order: 0,
            }),
            processing: None,
            unit: "%".to_string(),
            display_name: "SpO2".to_string(),
        }),
        _ => None
    }
}