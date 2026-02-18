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
    pub min_window_seconds: f32,
    pub preferred_window_seconds: f32,
    pub min_value: f32,
    pub max_value: f32,
    pub order: u8, 
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HrvMetric {
    Sdnn,
    Rmssd,
    LfHf,
    StressIndex,
    Pnn50,
    Sd1Sd2,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CalculationMethod {
    Rate(RateStrategy),
    HrvFromPeaks(HrvMetric),
    Average,
    BpSystolic,
    BpDiastolic,
    PulsePressure,
    IeRatio,
}

#[derive(Debug, Clone, Copy)]
pub enum PostProcessOp {
    None,
    Detrend,
    MovingAverage,
    Standardize,
    MovingAverageStandardize,
    DetrendMovingAverageStandardize,
}

#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    pub operation: PostProcessOp, 
    pub min_window_seconds: f32,
    pub min_freq: Option<f32>,
    pub max_freq: Option<f32>,
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

pub fn get_vital_meta(vital_id: &str) -> Option<VitalMeta> {   
    match vital_id {

        // PPG Waveform
        "ppg_waveform" | "ppg" => Some(VitalMeta {
            id: "ppg_waveform".to_string(),
            vital_type: VitalType::Provided,
            derivations: vec![],
            processing: Some(ProcessingConfig {
                operation: PostProcessOp::MovingAverageStandardize,
                min_window_seconds: 5.0,
                min_freq: Some(0.67),
                max_freq: Some(3.67),
            }),
            unit: "unitless".to_string(),
            display_name: "PPG Waveform".to_string(),
        }),

        // Respiratory Waveform
        "respiratory_waveform" | "resp_waveform" | "resp" => Some(VitalMeta {
            id: "respiratory_waveform".to_string(),
            vital_type: VitalType::Provided,
            derivations: vec![],
            processing: Some(ProcessingConfig {
                operation: PostProcessOp::MovingAverageStandardize,
                min_window_seconds: 10.0,
                min_freq: Some(0.05),
                max_freq: Some(1.0),
            }),
            unit: "unitless".to_string(),
            display_name: "RESP Waveform".to_string(),
        }),

        // Heart Rate
        "heart_rate" | "hr" | "pulse" => Some(VitalMeta {
            id: "heart_rate".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![DerivationConfig {
                source_signal: "ppg_waveform".to_string(),
                method: CalculationMethod::Rate(RateStrategy::Periodogram { 
                    target_res_hz: 0.005
                }),
                min_window_seconds: 5.0,
                preferred_window_seconds: 10.0,
                min_value: 40.0,
                max_value: 220.0,
                order: 0,
            }],
            processing: None,
            unit: "bpm".to_string(),
            display_name: "Heart Rate".to_string(),
        }),

        // Respiratory Rate
        "respiratory_rate" | "rr" | "resp_rate" => Some(VitalMeta {
            id: "respiratory_rate".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![DerivationConfig {
                source_signal: "respiratory_waveform".to_string(),
                method: CalculationMethod::Rate(RateStrategy::Periodogram { 
                    target_res_hz: 0.01
                }),
                min_window_seconds: 10.0,
                preferred_window_seconds: 30.0,
                min_value: 3.0,
                max_value: 60.0,
                order: 0,
            }],
            processing: None,
            unit: "bpm".to_string(),
            display_name: "Respiratory Rate".to_string(),
        }),

        // HRV (SDNN)
        "hrv_sdnn" | "sdnn" => Some(VitalMeta {
            id: "hrv_sdnn".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![DerivationConfig {
                source_signal: "ppg_waveform".to_string(),
                method: CalculationMethod::HrvFromPeaks(HrvMetric::Sdnn),
                min_window_seconds: 20.0,
                preferred_window_seconds: 120.0,
                min_value: 1.0,
                max_value: 200.0,
                order: 1,
            }],
            processing: None,
            unit: "ms".to_string(),
            display_name: "Heart Rate Variability (SDNN)".to_string(),
        }),

        // HRV (RMSSD)
        "hrv_rmssd" | "rmssd" => Some(VitalMeta {
            id: "hrv_rmssd".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![DerivationConfig {
                source_signal: "ppg_waveform".to_string(),
                method: CalculationMethod::HrvFromPeaks(HrvMetric::Rmssd),
                min_window_seconds: 20.0,
                preferred_window_seconds: 60.0,
                min_value: 1.0,
                max_value: 200.0,
                order: 1,
            }],
            processing: None,
            unit: "ms".to_string(),
            display_name: "Heart Rate Variability (RMSSD)".to_string(),
        }),

        // HRV (pNN50)
        "hrv_pnn50" | "pnn50" => Some(VitalMeta {
            id: "hrv_pnn50".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![DerivationConfig {
                source_signal: "ppg_waveform".to_string(),
                method: CalculationMethod::HrvFromPeaks(HrvMetric::Pnn50),
                min_window_seconds: 30.0,
                preferred_window_seconds: 60.0,
                min_value: 0.0,
                max_value: 100.0,
                order: 1,
            }],
            processing: None,
            unit: "%".to_string(),
            display_name: "Heart Rate Variability (pNN50)".to_string(),
        }),

        // HRV (LF/HF)
        "hrv_lfhf" | "lfhf" => Some(VitalMeta {
            id: "hrv_lfhf".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![DerivationConfig {
                source_signal: "ppg_waveform".to_string(),
                method: CalculationMethod::HrvFromPeaks(HrvMetric::LfHf),
                min_window_seconds: 55.0,
                preferred_window_seconds: 120.0,
                min_value: 0.0,
                max_value: 10.0,
                order: 1,
            }],
            processing: None,
            unit: "ratio".to_string(),
            display_name: "Heart Rate Variability (LF/HF)".to_string(),
        }),

        // HRV (SD1/SD2)
        "hrv_sd1sd2" | "sd1sd2" => Some(VitalMeta {
            id: "hrv_sd1sd2".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![DerivationConfig {
                source_signal: "ppg_waveform".to_string(),
                method: CalculationMethod::HrvFromPeaks(HrvMetric::Sd1Sd2),
                min_window_seconds: 55.0,
                preferred_window_seconds: 120.0,
                min_value: 0.0,
                max_value: 1.0,
                order: 1,
            }],
            processing: None,
            unit: "ratio".to_string(),
            display_name: "Heart Rate Variability (SD1/SD2)".to_string(),
        }),
        
        // I:E Ratio
        "ie_ratio" | "resp_ie" => Some(VitalMeta {
            id: "ie_ratio".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![DerivationConfig {
                source_signal: "respiratory_waveform".to_string(),
                method: CalculationMethod::IeRatio,
                min_window_seconds: 15.0,
                preferred_window_seconds: 30.0,
                min_value: 0.2,
                max_value: 5.0,
                order: 1,
            }],
            processing: None,
            unit: "ratio".to_string(),
            display_name: "I:E Ratio".to_string(),
        }),

        // Stress Index (SI)
        "stress_index" | "si" => Some(VitalMeta {
            id: "stress_index".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![DerivationConfig {
                source_signal: "ppg_waveform".to_string(),
                method: CalculationMethod::HrvFromPeaks(HrvMetric::StressIndex),
                min_window_seconds: 55.0,
                preferred_window_seconds: 120.0,
                min_value: 0.0,
                max_value: 1000.0,
                order: 1,
            }],
            processing: None,
            unit: "SI".to_string(),
            display_name: "Stress Index".to_string(),
        }),

        // SpO2
        "spo2" => Some(VitalMeta {
            id: "spo2".to_string(),
            vital_type: VitalType::Provided,
            derivations: vec![DerivationConfig {
                source_signal: "spo2".to_string(),
                method: CalculationMethod::Average,
                min_window_seconds: 1.0,
                preferred_window_seconds: 5.0,
                min_value: 70.0,
                max_value: 100.0,
                order: 0,
            }],
            processing: None,
            unit: "%".to_string(),
            display_name: "SpO2".to_string(),
        }),

        // Arterial Blood Pressure Waveform
        "abp_waveform" => Some(VitalMeta {
            id: "abp_waveform".to_string(),
            vital_type: VitalType::Provided,
            derivations: vec![],
            processing: Some(ProcessingConfig {
                operation: PostProcessOp::None,
                min_window_seconds: 5.0,
                min_freq: None,
                max_freq: None,
            }),
            unit: "mmHg".to_string(),
            display_name: "ABP Waveform".to_string(),
        }),

        // Systolic Blood Pressure
        "sbp" | "bp_sys" | "systolic" => Some(VitalMeta {
            id: "sbp".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![
                DerivationConfig {
                    source_signal: "abp_waveform".to_string(),
                    method: CalculationMethod::BpSystolic, 
                    min_window_seconds: 5.0,
                    preferred_window_seconds: 10.0,
                    min_value: 60.0,
                    max_value: 200.0,
                    order: 1,
                },
                DerivationConfig {
                    source_signal: "sbp".to_string(),
                    method: CalculationMethod::Average,
                    min_window_seconds: 5.0,
                    preferred_window_seconds: 10.0,
                    min_value: 60.0,
                    max_value: 200.0,
                    order: 1,
                },
            ],
            processing: None,
            unit: "mmHg".to_string(),
            display_name: "Systolic Blood Pressure".to_string(),
        }),

        // Diastolic Blood Pressure
        "dbp" | "bp_dia" | "diastolic" => Some(VitalMeta {
            id: "dbp".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![
                DerivationConfig {
                    source_signal: "abp_waveform".to_string(),
                    method: CalculationMethod::BpDiastolic, 
                    min_window_seconds: 5.0,
                    preferred_window_seconds: 10.0,
                    min_value: 40.0,
                    max_value: 120.0,
                    order: 1,
                },
                DerivationConfig {
                    source_signal: "dbp".to_string(),
                    method: CalculationMethod::Average,
                    min_window_seconds: 5.0,
                    preferred_window_seconds: 10.0,
                    min_value: 40.0,
                    max_value: 120.0,
                    order: 1,
                },
            ],
            processing: None,
            unit: "mmHg".to_string(),
            display_name: "Diastolic Blood Pressure".to_string(),
        }),

        // Mean Arterial Pressure (MAP)
        "map" | "mean_arterial_pressure" => Some(VitalMeta {
            id: "map".to_string(),
            vital_type: VitalType::Derived,
            derivations: vec![DerivationConfig {
                source_signal: "abp_waveform".to_string(),
                method: CalculationMethod::Average,
                min_window_seconds: 5.0,
                preferred_window_seconds: 10.0,
                min_value: 40.0,
                max_value: 200.0,
                order: 1,
            }],
            processing: None,
            unit: "mmHg".to_string(),
            display_name: "Mean Arterial Pressure".to_string(),
        }),

        // Pulse Pressure (PP)
        "pp" | "pulse_pressure" => Some(VitalMeta {
            id: "pulse_pressure".to_string(),
            vital_type: VitalType::Derived,
            // TODO: Could add derivation from SBP + DBP if we add multi-source-signal support
            derivations: vec![DerivationConfig {
                source_signal: "abp_waveform".to_string(),
                method: CalculationMethod::PulsePressure,
                min_window_seconds: 5.0,
                preferred_window_seconds: 10.0,
                min_value: 10.0,
                max_value: 100.0,
                order: 1,
            }],
            processing: None,
            unit: "mmHg".to_string(),
            display_name: "Pulse Pressure".to_string(),
        }),

        _ => None
    }
}