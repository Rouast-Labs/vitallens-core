use crate::signal::rate::RateStrategy;

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HrvMetric {
    Sdnn,
    Rmssd,
    LfHf,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CalculationMethod {
    Rate(RateStrategy),
    HrvFromPeaks(HrvMetric),
    Average,
    BpSystolic,
    BpDiastolic,
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
    pub derivations: Vec<DerivationConfig>,
    pub processing: Option<ProcessingConfig>,
    pub unit: String,
    pub display_name: String,
}

// --- THE REGISTRY ---

// TODO: What other vitals could we support given our provided signals?

pub fn get_vital_meta(vital_id: &str) -> Option<VitalMeta> {   
    match vital_id {
        "ppg_waveform" | "ppg" => Some(VitalMeta {
            id: "ppg_waveform".to_string(),
            vital_type: VitalType::Provided,
            derivations: vec![],
            processing: Some(ProcessingConfig {
                operation: PostProcessOp::Detrend,
                min_window_seconds: 5.0,
            }),
            unit: "unitless".to_string(),
            display_name: "PPG Waveform".to_string(),
        }),
        "respiratory_waveform" | "resp_waveform" | "resp" => Some(VitalMeta {
            id: "respiratory_waveform".to_string(),
            vital_type: VitalType::Provided,
            derivations: vec![],
            processing: Some(ProcessingConfig {
                operation: PostProcessOp::Detrend,
                min_window_seconds: 10.0,
            }),
            unit: "unitless".to_string(),
            display_name: "RESP Waveform".to_string(),
        }),
        "heart_rate" | "hr" | "pulse" => Some(VitalMeta {
            id: "heart_rate".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![DerivationConfig {
                source_signal: "ppg_waveform".to_string(),
                method: CalculationMethod::Rate(RateStrategy::Periodogram { 
                    target_res_hz: 0.005
                }),
                min_required_seconds: 5.0,
                optimal_window_seconds: 10.0,
                min_value: 40.0,
                max_value: 240.0,
                order: 0,
            }],
            processing: None,
            unit: "bpm".to_string(),
            display_name: "Heart Rate".to_string(),
        }),
        "respiratory_rate" | "rr" | "resp_rate" => Some(VitalMeta {
            id: "respiratory_rate".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![DerivationConfig {
                source_signal: "respiratory_waveform".to_string(),
                method: CalculationMethod::Rate(RateStrategy::Periodogram { 
                    target_res_hz: 0.01
                }),
                min_required_seconds: 10.0,
                optimal_window_seconds: 30.0,
                min_value: 4.0,
                max_value: 60.0,
                order: 0,
            }],
            processing: None,
            unit: "bpm".to_string(),
            display_name: "Respiratory Rate".to_string(),
        }),
        "hrv_sdnn" | "sdnn" => Some(VitalMeta {
            id: "hrv_sdnn".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![DerivationConfig {
                source_signal: "ppg_waveform".to_string(),
                method: CalculationMethod::HrvFromPeaks(HrvMetric::Sdnn),
                min_required_seconds: 20.0,
                optimal_window_seconds: 60.0,
                min_value: 1.0,
                max_value: 200.0,
                order: 1,
            }],
            processing: None,
            unit: "ms".to_string(),
            display_name: "Heart Rate Variability (SDNN)".to_string(),
        }),
        "hrv_rmssd" | "rmssd" => Some(VitalMeta {
            id: "hrv_rmssd".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![DerivationConfig {
                source_signal: "ppg_waveform".to_string(),
                method: CalculationMethod::HrvFromPeaks(HrvMetric::Rmssd),
                min_required_seconds: 20.0,
                optimal_window_seconds: 60.0,
                min_value: 1.0,
                max_value: 200.0,
                order: 1,
            }],
            processing: None,
            unit: "ms".to_string(),
            display_name: "Heart Rate Variability (RMSSD)".to_string(),
        }),
        "hrv_lfhf" | "lfhf" => Some(VitalMeta {
            id: "hrv_lfhf".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![DerivationConfig {
                source_signal: "ppg_waveform".to_string(),
                method: CalculationMethod::HrvFromPeaks(HrvMetric::LfHf),
                min_required_seconds: 55.0,
                optimal_window_seconds: 120.0,
                min_value: 0.0,
                max_value: 10.0,
                order: 1,
            }],
            processing: None,
            unit: "ratio".to_string(),
            display_name: "Heart Rate Variability (LF/HF)".to_string(),
        }),
        "spo2" => Some(VitalMeta {
            id: "spo2".to_string(),
            vital_type: VitalType::Provided,
            derivations: vec![DerivationConfig {
                source_signal: "spo2".to_string(),
                method: CalculationMethod::Average,
                min_required_seconds: 1.0,
                optimal_window_seconds: 5.0,
                min_value: 70.0,
                max_value: 100.0,
                order: 0,
            }],
            processing: None,
            unit: "%".to_string(),
            display_name: "SpO2".to_string(),
        }),
        "abp_waveform" => Some(VitalMeta {
            id: "abp_waveform".to_string(),
            vital_type: VitalType::Provided,
            derivations: vec![],
            processing: Some(ProcessingConfig {
                operation: PostProcessOp::None,
                min_window_seconds: 5.0,
            }),
            unit: "mmHg".to_string(),
            display_name: "ABP Waveform".to_string(),
        }),
        "sbp" | "bp_sys" | "systolic" => Some(VitalMeta {
            id: "sbp".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![
                DerivationConfig {
                    source_signal: "abp_waveform".to_string(),
                    method: CalculationMethod::BpSystolic, 
                    min_required_seconds: 5.0,
                    optimal_window_seconds: 10.0,
                    min_value: 60.0,
                    max_value: 200.0,
                    order: 1,
                },
                DerivationConfig {
                    source_signal: "sbp".to_string(),
                    method: CalculationMethod::Average,
                    min_required_seconds: 5.0,
                    optimal_window_seconds: 10.0,
                    min_value: 60.0,
                    max_value: 200.0,
                    order: 1,
                },
            ],
            processing: None,
            unit: "mmHg".to_string(),
            display_name: "Systolic Blood Pressure".to_string(),
        }),
        "dbp" | "bp_dia" | "diastolic" => Some(VitalMeta {
            id: "dbp".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![
                DerivationConfig {
                    source_signal: "abp_waveform".to_string(),
                    method: CalculationMethod::BpDiastolic, 
                    min_required_seconds: 5.0,
                    optimal_window_seconds: 10.0,
                    min_value: 40.0,
                    max_value: 120.0,
                    order: 1,
                },
                DerivationConfig {
                    source_signal: "dbp".to_string(),
                    method: CalculationMethod::Average,
                    min_required_seconds: 5.0,
                    optimal_window_seconds: 10.0,
                    min_value: 40.0,
                    max_value: 120.0,
                    order: 1,
                },
            ],
            processing: None,
            unit: "mmHg".to_string(),
            display_name: "Diastolic Blood Pressure".to_string(),
        }),
        _ => None
    }
}