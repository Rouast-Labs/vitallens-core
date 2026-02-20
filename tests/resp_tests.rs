use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use serde::Deserialize;
use test_generator::test_resources;

use vitallens_core::{Session, SessionConfig, SessionInput, WaveformMode};
use vitallens_core::registry;

const TOLERANCE_IE_RATIO: f32 = 0.15;

// --- LOGGER SETUP ---
fn init_logger() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[derive(Deserialize, Debug)]
struct ReferenceData {
    vital_signs: Vitals,
    fps: f32, 
}

#[derive(Deserialize, Debug)]
struct Vitals {
    respiratory_waveform: Option<Waveform>,
    ie_ratio: Option<Vital>,
}

#[derive(Deserialize, Debug)]
struct Waveform {
    data: Vec<f32>,
    confidence: Option<Vec<f32>>,
}

#[derive(Deserialize, Debug)]
struct Vital {
    value: f32,
    confidence: f32,
}

#[test_resources("tests/fixtures/*.json")]
fn test_ie_ratio_accuracy(resource: &str) {
    init_logger();

    let path = Path::new(resource);
    let filename = path.file_name().unwrap().to_str().unwrap();
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);
    let ref_data: ReferenceData = serde_json::from_reader(reader).expect("Failed to parse JSON");

    if let (Some(gt), Some(resp)) = (ref_data.vital_signs.ie_ratio, ref_data.vital_signs.respiratory_waveform) {
        let meta = registry::get_vital_meta("ie_ratio").unwrap();
        let min_win = meta.derivations[0].min_window_seconds;
        let duration_sec = resp.data.len() as f32 / ref_data.fps;

        if duration_sec < min_win {
            println!(" [SKIP] {} - insufficient duration ({:.1}s < {:.1}s)", filename, duration_sec, min_win);
            return;
        }

        println!("=== TEST IE RATIO: {} ===", filename);

        let config = SessionConfig {
            supported_vitals: vec!["respiratory_rate".to_string(), "ie_ratio".to_string()],
            return_waveforms: None,
            fps_target: ref_data.fps,
            input_size: 30,
            n_inputs: 4,
            roi_method: "face".to_string(),
        };
        
        let session = Session::new(config);
        let conf = resp.confidence.unwrap_or_else(|| vec![1.0; resp.data.len()]);

        let input = SessionInput {
            timestamp: (0..resp.data.len()).map(|t| t as f64 / ref_data.fps as f64).collect(),
            signals: [("respiratory_waveform".to_string(), vitallens_core::types::SignalInput { 
                data: resp.data, 
                confidence: conf 
            })].into(),
            face: None,
        };

        let result = session.process(input, WaveformMode::Global);

        if let Some(res) = result.vitals.get("ie_ratio") {
             
            let calc_conf = res.confidence;
            
            let val_diff = (res.value - gt.value).abs();
            let conf_diff = (calc_conf - gt.confidence).abs();

            println!(" -> Calculated: {:.3} (Conf: {:.2}), Ref: {:.3} (Ref Conf: {:.2})", 
                res.value, calc_conf, gt.value, gt.confidence);

            assert!(val_diff <= TOLERANCE_IE_RATIO, 
                "IE Ratio value mismatch in {}: got {}, ref {}", filename, res.value, gt.value);
                
            assert!(conf_diff <= 0.1, 
                "IE Ratio confidence mismatch in {}: got {}, ref {}", filename, calc_conf, gt.confidence);
        } else {
            panic!("IE Ratio returned None for {}", filename);
        }
    }
}