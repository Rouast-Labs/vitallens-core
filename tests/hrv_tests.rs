use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use serde::Deserialize;
use test_generator::test_resources;

use vitallens_core::signal::{peaks, hrv};
use vitallens_core::registry::{self, HrvMetric};

// Tolerance for HRV metrics
const SDNN_TOLERANCE_MS: f32 = 5.0;
const RMSSD_TOLERANCE_MS: f32 = 5.0;
const LFHF_TOLERANCE: f32 = 0.5;
const PNN50_TOLERANCE: f32 = 5.0;
const SD1SD2_TOLERANCE: f32 = 0.05;
const STRESS_INDEX_TOLERANCE: f32 = 20.0;

#[derive(Deserialize, Debug)]
struct ReferenceData {
    vital_signs: Vitals,
    fps: f32, 
}

#[derive(Deserialize, Debug)]
struct Vitals {
    ppg_waveform: Option<Waveform>,
    heart_rate: Option<Vital>,
    hrv_sdnn: Option<Vital>,
    hrv_rmssd: Option<Vital>,
    hrv_lfhf: Option<Vital>,
    hrv_pnn50: Option<Vital>,
    hrv_sd1sd2: Option<Vital>,
    stress_index: Option<Vital>,
}

#[derive(Deserialize, Debug)]
struct Waveform {
    data: Vec<f32>,
    confidence: Option<Vec<f32>>, 
    #[allow(dead_code)]
    peak_indices: Option<Vec<usize>>, 
}

#[derive(Deserialize, Debug)]
struct Vital {
    value: f32,
}

// --- Verification Helpers ---

#[allow(clippy::too_many_arguments)]
fn verify_sdnn(
    filename: &str,
    fs: f32,
    signal: &[f32], 
    confidence_source: Option<&Vec<f32>>,
    ground_truth_sdnn: f32,
    rate_hint: Option<f32>
) -> Result<(), String> {
    
    let meta_hr = registry::get_vital_meta("heart_rate").unwrap();
    let deriv_hr = &meta_hr.derivations[0];
    
    let meta_sdnn = registry::get_vital_meta("hrv_sdnn").unwrap();
    let deriv_sdnn = &meta_sdnn.derivations[0];

    let duration_sec = signal.len() as f32 / fs;
    if duration_sec < deriv_sdnn.min_window_seconds {
        println!("[{}] SDNN SKIPPED: Duration {:.1}s < Min Window {:.1}s", 
            filename, duration_sec, deriv_sdnn.min_window_seconds);
        return Ok(());
    }

    let confidence = confidence_source.cloned().unwrap_or_else(|| vec![1.0; signal.len()]);

    let timestamps: Vec<f64> = (0..signal.len()).map(|i| i as f64 / fs as f64).collect();
    let bounds = peaks::SignalBounds { 
        min_rate: deriv_hr.min_value, 
        max_rate: deriv_hr.max_value 
    };

    let (calculated, calc_conf) = hrv::estimate_hrv(
        signal, fs, HrvMetric::Sdnn, &timestamps, &confidence, bounds, rate_hint
    );

    let diff = (calculated - ground_truth_sdnn).abs();
    let max_input_conf = confidence.iter().fold(0.0f32, |a, &b| a.max(b));
    
    println!("[{}] SDNN: Calc={:.2}ms (Conf {:.2}), Ref={:.2}ms, Diff={:.2}ms", filename, calculated, calc_conf, ground_truth_sdnn, diff);

    if calc_conf < 0.0 || calc_conf > max_input_conf + f32::EPSILON {
        return Err(format!("[{}] SDNN Confidence Invalid. Got {:.2}, Max Input {:.2}", filename, calc_conf, max_input_conf));
    }

    if diff <= SDNN_TOLERANCE_MS {
        Ok(())
    } else {
        Err(format!("[{}] SDNN Mismatch. Expected {:.2}, Got {:.2} (Diff {:.2})", filename, ground_truth_sdnn, calculated, diff))
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_rmssd(
    filename: &str,
    fs: f32,
    signal: &[f32], 
    confidence_source: Option<&Vec<f32>>,
    ground_truth_rmssd: f32,
    rate_hint: Option<f32>
) -> Result<(), String> {
    
    let meta_hr = registry::get_vital_meta("heart_rate").unwrap();
    let deriv_hr = &meta_hr.derivations[0];
    
    let meta_rmssd = registry::get_vital_meta("hrv_rmssd").unwrap();
    let deriv_rmssd = &meta_rmssd.derivations[0];

    let duration_sec = signal.len() as f32 / fs;
    if duration_sec < deriv_rmssd.min_window_seconds {
        println!("[{}] RMSSD SKIPPED: Duration {:.1}s < Min Window {:.1}s", 
            filename, duration_sec, deriv_rmssd.min_window_seconds);
        return Ok(());
    }

    let confidence = confidence_source.cloned().unwrap_or_else(|| vec![1.0; signal.len()]);

    let timestamps: Vec<f64> = (0..signal.len()).map(|i| i as f64 / fs as f64).collect();
    let bounds = peaks::SignalBounds { 
        min_rate: deriv_hr.min_value, 
        max_rate: deriv_hr.max_value 
    };

    let (calculated, calc_conf) = hrv::estimate_hrv(
        signal, fs, HrvMetric::Rmssd, &timestamps, &confidence, bounds, rate_hint
    );

    let diff = (calculated - ground_truth_rmssd).abs();
    let max_input_conf = confidence.iter().fold(0.0f32, |a, &b| a.max(b));
    
    println!("[{}] RMSSD: Calc={:.2}ms (Conf {:.2}), Ref={:.2}ms, Diff={:.2}ms", filename, calculated, calc_conf, ground_truth_rmssd, diff);

    if calc_conf < 0.0 || calc_conf > max_input_conf + f32::EPSILON {
        return Err(format!("[{}] RMSSD Confidence Invalid. Got {:.2}, Max Input {:.2}", filename, calc_conf, max_input_conf));
    }

    if diff <= RMSSD_TOLERANCE_MS {
        Ok(())
    } else {
        Err(format!("[{}] RMSSD Mismatch. Expected {:.2}, Got {:.2} (Diff {:.2})", filename, ground_truth_rmssd, calculated, diff))
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_lfhf(
    filename: &str,
    fs: f32,
    signal: &[f32], 
    confidence_source: Option<&Vec<f32>>,
    ground_truth_lfhf: f32,
    rate_hint: Option<f32>
) -> Result<(), String> {
    
    let meta_hr = registry::get_vital_meta("heart_rate").unwrap();
    let deriv_hr = &meta_hr.derivations[0];
    
    let meta_lfhf = registry::get_vital_meta("hrv_lfhf").unwrap();
    let deriv_lfhf = &meta_lfhf.derivations[0];

    let duration_sec = signal.len() as f32 / fs;
    if duration_sec < deriv_lfhf.min_window_seconds {
        println!("[{}] LF/HF SKIPPED: Duration {:.1}s < Min Window {:.1}s", 
            filename, duration_sec, deriv_lfhf.min_window_seconds);
        return Ok(());
    }

    let confidence = confidence_source.cloned().unwrap_or_else(|| vec![1.0; signal.len()]);

    let timestamps: Vec<f64> = (0..signal.len()).map(|i| i as f64 / fs as f64).collect();
    let bounds = peaks::SignalBounds { 
        min_rate: deriv_hr.min_value, 
        max_rate: deriv_hr.max_value 
    };

    let (calculated, calc_conf) = hrv::estimate_hrv(
        signal, fs, HrvMetric::LfHf, &timestamps, &confidence, bounds, rate_hint
    );

    let diff = (calculated - ground_truth_lfhf).abs();
    let max_input_conf = confidence.iter().fold(0.0f32, |a, &b| a.max(b));
    
    println!("[{}] LF/HF: Calc={:.2} (Conf {:.2}), Ref={:.2}, Diff={:.2}", filename, calculated, calc_conf, ground_truth_lfhf, diff);

    if calc_conf < 0.0 || calc_conf > max_input_conf + f32::EPSILON {
        return Err(format!("[{}] LF/HF Confidence Invalid. Got {:.2}, Max Input {:.2}", filename, calc_conf, max_input_conf));
    }

    if diff <= LFHF_TOLERANCE {
        Ok(())
    } else {
        Err(format!("[{}] LF/HF Mismatch. Expected {:.2}, Got {:.2} (Diff {:.2})", filename, ground_truth_lfhf, calculated, diff))
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_stress_index(
    filename: &str,
    fs: f32,
    signal: &[f32], 
    confidence_source: Option<&Vec<f32>>,
    ground_truth_si: f32,
    rate_hint: Option<f32>
) -> Result<(), String> {
    
    let meta_hr = registry::get_vital_meta("heart_rate").unwrap();
    let deriv_hr = &meta_hr.derivations[0];
    
    let meta_si = registry::get_vital_meta("stress_index").unwrap();
    let deriv_si = &meta_si.derivations[0];

    let duration_sec = signal.len() as f32 / fs;
    if duration_sec < deriv_si.min_window_seconds {
        println!("[{}] STRESS INDEX SKIPPED: Duration {:.1}s < Min Window {:.1}s", 
            filename, duration_sec, deriv_si.min_window_seconds);
        return Ok(());
    }

    let confidence = confidence_source.cloned().unwrap_or_else(|| vec![1.0; signal.len()]);

    let timestamps: Vec<f64> = (0..signal.len()).map(|i| i as f64 / fs as f64).collect();
    let bounds = peaks::SignalBounds { 
        min_rate: deriv_hr.min_value, 
        max_rate: deriv_hr.max_value 
    };

    let (calculated, calc_conf) = hrv::estimate_hrv(
        signal, fs, HrvMetric::StressIndex, &timestamps, &confidence, bounds, rate_hint
    );

    let diff = (calculated - ground_truth_si).abs();
    let max_input_conf = confidence.iter().fold(0.0f32, |a, &b| a.max(b));
    
    println!("[{}] Stress Index: Calc={:.2} (Conf {:.2}), Ref={:.2}, Diff={:.2}", filename, calculated, calc_conf, ground_truth_si, diff);

    if calc_conf < 0.0 || calc_conf > max_input_conf + f32::EPSILON {
        return Err(format!("[{}] Stress Index Confidence Invalid. Got {:.2}, Max Input {:.2}", filename, calc_conf, max_input_conf));
    }

    if diff <= STRESS_INDEX_TOLERANCE {
        Ok(())
    } else {
        Err(format!("[{}] Stress Index Mismatch. Expected {:.2}, Got {:.2} (Diff {:.2})", filename, ground_truth_si, calculated, diff))
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_pnn50(
    filename: &str,
    fs: f32,
    signal: &[f32], 
    confidence_source: Option<&Vec<f32>>,
    ground_truth_pnn50: f32,
    rate_hint: Option<f32>
) -> Result<(), String> {
    
    let meta_hr = registry::get_vital_meta("heart_rate").unwrap();
    let deriv_hr = &meta_hr.derivations[0];
    
    let meta_pnn50 = registry::get_vital_meta("hrv_pnn50").unwrap();
    let deriv_pnn50 = &meta_pnn50.derivations[0];

    let duration_sec = signal.len() as f32 / fs;
    if duration_sec < deriv_pnn50.min_window_seconds {
        println!("[{}] pNN50 SKIPPED: Duration {:.1}s < Min Window {:.1}s", 
            filename, duration_sec, deriv_pnn50.min_window_seconds);
        return Ok(());
    }

    let confidence = confidence_source.cloned().unwrap_or_else(|| vec![1.0; signal.len()]);

    let timestamps: Vec<f64> = (0..signal.len()).map(|i| i as f64 / fs as f64).collect();
    let bounds = peaks::SignalBounds { 
        min_rate: deriv_hr.min_value, 
        max_rate: deriv_hr.max_value 
    };

    let (calculated, calc_conf) = hrv::estimate_hrv(
        signal, fs, HrvMetric::Pnn50, &timestamps, &confidence, bounds, rate_hint
    );

    let diff = (calculated - ground_truth_pnn50).abs();
    let max_input_conf = confidence.iter().fold(0.0f32, |a, &b| a.max(b));
    
    println!("[{}] pNN50: Calc={:.2}% (Conf {:.2}), Ref={:.2}%, Diff={:.2}%", filename, calculated, calc_conf, ground_truth_pnn50, diff);

    if calc_conf < 0.0 || calc_conf > max_input_conf + f32::EPSILON {
        return Err(format!("[{}] pNN50 Confidence Invalid. Got {:.2}, Max Input {:.2}", filename, calc_conf, max_input_conf));
    }

    if diff <= PNN50_TOLERANCE {
        Ok(())
    } else {
        Err(format!("[{}] pNN50 Mismatch. Expected {:.2}, Got {:.2} (Diff {:.2})", filename, ground_truth_pnn50, calculated, diff))
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_sd1sd2(
    filename: &str,
    fs: f32,
    signal: &[f32], 
    confidence_source: Option<&Vec<f32>>,
    ground_truth_ratio: f32,
    rate_hint: Option<f32>
) -> Result<(), String> {
    
    let meta_hr = registry::get_vital_meta("heart_rate").unwrap();
    let deriv_hr = &meta_hr.derivations[0];
    
    let meta_s = registry::get_vital_meta("hrv_sd1sd2").unwrap();
    let deriv_s = &meta_s.derivations[0];

    let duration_sec = signal.len() as f32 / fs;
    if duration_sec < deriv_s.min_window_seconds {
        println!("[{}] SD1/SD2 SKIPPED: Duration {:.1}s < Min Window {:.1}s", 
            filename, duration_sec, deriv_s.min_window_seconds);
        return Ok(());
    }

    let confidence = confidence_source.cloned().unwrap_or_else(|| vec![1.0; signal.len()]);

    let timestamps: Vec<f64> = (0..signal.len()).map(|i| i as f64 / fs as f64).collect();
    let bounds = peaks::SignalBounds { 
        min_rate: deriv_hr.min_value, 
        max_rate: deriv_hr.max_value 
    };

    let (calculated, calc_conf) = hrv::estimate_hrv(
        signal, fs, HrvMetric::Sd1Sd2, &timestamps, &confidence, bounds, rate_hint
    );

    let diff = (calculated - ground_truth_ratio).abs();
    let max_input_conf = confidence.iter().fold(0.0f32, |a, &b| a.max(b));
    
    println!("[{}] SD1/SD2: Calc={:.3} (Conf {:.2}), Ref={:.3}, Diff={:.3}", filename, calculated, calc_conf, ground_truth_ratio, diff);

    if calc_conf < 0.0 || calc_conf > max_input_conf + f32::EPSILON {
        return Err(format!("[{}] SD1/SD2 Confidence Invalid. Got {:.2}, Max Input {:.2}", filename, calc_conf, max_input_conf));
    }

    if diff <= SD1SD2_TOLERANCE {
        Ok(())
    } else {
        Err(format!("[{}] SD1/SD2 Mismatch. Expected {:.3}, Got {:.3} (Diff {:.3})", filename, ground_truth_ratio, calculated, diff))
    }
}

// --- Main Test Runner ---

#[test_resources("tests/fixtures/*.json")]
fn test_hrv_integrity(resource: &str) {
    let path = Path::new(resource);
    let filename = path.file_name().unwrap().to_str().unwrap();
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);
    
    let ref_data: ReferenceData = serde_json::from_reader(reader)
        .expect("Failed to parse JSON.");
    let fs = ref_data.fps;
    
    let mut failures = Vec::new();

    // Common Data
    let rate_hint = ref_data.vital_signs.heart_rate.as_ref().map(|hr| hr.value);
    
    if let Some(ppg) = &ref_data.vital_signs.ppg_waveform {
        
        // 1. Verify SDNN
        if let Some(sdnn_ref) = &ref_data.vital_signs.hrv_sdnn {
            if let Err(e) = verify_sdnn(
                filename, fs, &ppg.data, ppg.confidence.as_ref(), sdnn_ref.value, rate_hint
            ) {
                failures.push(e);
            }
        }

        // 2. Verify RMSSD
        if let Some(rmssd_ref) = &ref_data.vital_signs.hrv_rmssd {
            if let Err(e) = verify_rmssd(
                filename, fs, &ppg.data, ppg.confidence.as_ref(), rmssd_ref.value, rate_hint
            ) {
                failures.push(e);
            }
        }

        // 3. Verify LF/HF
        if let Some(lfhf_ref) = &ref_data.vital_signs.hrv_lfhf {
            if let Err(e) = verify_lfhf(
                filename, fs, &ppg.data, ppg.confidence.as_ref(), lfhf_ref.value, rate_hint
            ) {
                failures.push(e);
            }
        }

        // 4. Verify pNN50
        if let Some(pnn50_ref) = &ref_data.vital_signs.hrv_pnn50 {
            if let Err(e) = verify_pnn50(
                filename, fs, &ppg.data, ppg.confidence.as_ref(), pnn50_ref.value, rate_hint
            ) {
                failures.push(e);
            }
        }

        // 5. Verify SD1/SD2
        if let Some(sd1sd2_ref) = &ref_data.vital_signs.hrv_sd1sd2 {
            if let Err(e) = verify_sd1sd2(
                filename, fs, &ppg.data, ppg.confidence.as_ref(), sd1sd2_ref.value, rate_hint
            ) {
                failures.push(e);
            }
        }
        
        // 6. Verify Stress Index
        if let Some(si_ref) = &ref_data.vital_signs.stress_index {
            if let Err(e) = verify_stress_index(
                filename, fs, &ppg.data, ppg.confidence.as_ref(), si_ref.value, rate_hint
            ) {
                failures.push(e);
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "\n\nTest failed for file: {:?}\n==================================================\n{}\n", 
            filename,
            failures.join("\n\n")
        );
    }
}