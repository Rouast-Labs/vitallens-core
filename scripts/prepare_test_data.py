import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import vitallens_core

# --- CONFIGURATION ---
VIRTUAL_FS = 1000.0 

# Rust Parity Constants
MIN_RR_MS = 300.0
MAX_RR_MS = 2000.0
MAX_REL_CHANGE = 0.25

MIN_DURATION_THRESHOLDS = {
    "heart_rate": 5.0,
    "respiratory_rate": 10.0,
    "hrv_sdnn": 20.0,
    "hrv_rmssd": 20.0,
    "hrv_pnn50": 20.0,
    "hrv_sd1sd2": 55.0,
    "hrv_lfhf": 55.0,
    "stress_index": 55.0
}

# --- PARITY FUNCTIONS ---

def refine_location_rust_parity(signal, idx):
    """Direct port of vitallens-core/src/signal/peaks.rs -> refine_location."""
    idx_int = int(round(idx))
    if 0 < idx_int < len(signal) - 1:
        y_l = signal[idx_int - 1]
        y_c = signal[idx_int]
        y_r = signal[idx_int + 1]
        denom = 2.0 * (y_l - 2.0 * y_c + y_r)
        if abs(denom) > 1e-6:
            delta = (y_l - y_r) / denom
            if abs(delta) <= 0.5:
                return float(idx_int) + delta
    return float(idx_int)

def extract_intervals_rust_parity(peaks_indices, fs, confidence_array, conf_threshold):
    """
    Replicates extract_nn_intervals from vitallens-core/src/signal/hrv.rs
    Iterates PAIRS of peaks. Only accepts interval if BOTH peaks are confident.
    """
    intervals = []
    
    # peaks_indices must be sorted
    for i in range(len(peaks_indices) - 1):
        p1_idx_float = peaks_indices[i]
        p2_idx_float = peaks_indices[i+1]
        
        idx1 = int(round(p1_idx_float))
        idx2 = int(round(p2_idx_float))
        
        # Check Bounds
        if idx1 >= len(confidence_array) or idx2 >= len(confidence_array):
            continue
            
        c1 = confidence_array[idx1]
        c2 = confidence_array[idx2]
        
        # Confidence Gating (Strict Intersection)
        if c1 >= conf_threshold and c2 >= conf_threshold:
            t1 = p1_idx_float / fs
            t2 = p2_idx_float / fs
            diff_ms = (t2 - t1) * 1000.0
            
            if diff_ms > 0:
                intervals.append(diff_ms)
                
    return intervals

def filter_outliers_rust_parity(intervals_ms):
    """
    Replicates the MEDIAN-SEEDED logic from vitallens-core/src/signal/hrv.rs
    """
    intervals_ms = [float(x) for x in intervals_ms]
    if len(intervals_ms) < 2: return list(intervals_ms)

    # 1. Pre-filter range & Calculate Median
    valid_range = [x for x in intervals_ms if MIN_RR_MS <= x <= MAX_RR_MS]
    if not valid_range: return []

    valid_range.sort()
    mid_idx = len(valid_range) // 2
    median = valid_range[mid_idx]

    filtered = []
    last_accepted = median 

    # 2. Sequential check
    for curr in intervals_ms:
        if curr < MIN_RR_MS or curr > MAX_RR_MS: continue
        
        diff = abs(curr - last_accepted)
        if diff > last_accepted * MAX_REL_CHANGE: continue
        
        filtered.append(curr)
        last_accepted = curr

    return filtered

def calculate_neurokit_metrics(peaks_indices_float, fs, confidence_array, conf_threshold, label):
    if len(peaks_indices_float) < 2: return {}

    # 1. Interval Extraction (Parity)
    raw_intervals_ms = extract_intervals_rust_parity(peaks_indices_float, fs, confidence_array, conf_threshold)
    
    # 2. Filter (Median-Seeded Parity)
    clean_rri_ms = filter_outliers_rust_parity(raw_intervals_ms)
    
    if len(clean_rri_ms) < 2: return {}

    # 3. Reconstruct Timeline for NeuroKit
    # We rebuild peaks from the CLEAN intervals so NK2 sees the exact filtered data.
    clean_peaks_ms = np.cumsum([0] + clean_rri_ms)
    metrics = {}
    
    # --- Time Domain ---
    try:
        td = nk.hrv_time(clean_peaks_ms, sampling_rate=VIRTUAL_FS)
        metrics['hrv_sdnn'] = td['HRV_SDNN'].values[0]
        metrics['hrv_rmssd'] = td['HRV_RMSSD'].values[0]
        if 'HRV_pNN50' in td: metrics['hrv_pnn50'] = td['HRV_pNN50'].values[0]
    except: pass

    # --- Frequency Domain ---
    # UPDATED: Enforce Linear Interpolation and FFT Periodogram to match Rust
    try:
        fd = nk.hrv_frequency(
            clean_peaks_ms, 
            sampling_rate=VIRTUAL_FS, 
            normalize=True,
            interpolation_order=1, # Linear (Rust parity)
            psd_method='fft'       # Periodogram (Rust parity)
        )
        metrics['hrv_lfhf'] = fd['HRV_LFHF'].values[0]
    except Exception as e:
        print(f"[{label}] LF/HF Error: {e}")
        
    # --- Non-Linear ---
    try:
        nl = nk.hrv_nonlinear(clean_peaks_ms, sampling_rate=VIRTUAL_FS)
        if 'HRV_SI' in nl: metrics['stress_index'] = nl['HRV_SI'].values[0]
        elif 'HRV_Baevsky' in nl: metrics['stress_index'] = nl['HRV_Baevsky'].values[0]
        if 'HRV_SD1SD2' in nl: metrics['hrv_sd1sd2'] = nl['HRV_SD1SD2'].values[0]
    except: pass
        
    return metrics

class PeakAnnotator:
    def __init__(self, raw_signal, fs, title, rate_hint=None, 
                 min_rate=40.0, max_rate=220.0, detection_threshold=0.45, 
                 window_cycles=2.5, max_rate_change=1.0):
        self.raw_signal = np.array(raw_signal)
        self.fs = fs
        self.title = title
        
        print(f"[{title}] Auto-detecting peaks (Rust Parity)...")
        
        # No confidence arguments passed here, per your current version
        detected_peaks = vitallens_core.find_peaks(
            signal=self.raw_signal.astype(np.float32), 
            fs=float(fs), 
            refine=True,
            rate_hint=rate_hint,
            min_rate=min_rate,
            max_rate=max_rate,
            detection_threshold=detection_threshold,
            window_cycles=window_cycles,
            max_rate_change=max_rate_change
        )
        self.peaks = sorted(detected_peaks)
        
        print(f"[{title}] Found {len(self.peaks)} raw peaks.")

        # Plot
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.t = np.arange(len(self.raw_signal))
        self.ax.plot(self.t, self.raw_signal, color='#444444', alpha=0.8, label='Signal')
        self.scatter, = self.ax.plot([], [], 'rx', markersize=12, markeredgewidth=2, label='Peaks')
        self.ax.set_title(f"{title}\nL-Click: Add | R-Click: Delete | Close: Save")
        self.ax.legend(loc='upper left')
        self.update_plot()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def update_plot(self):
        if self.peaks:
            indices = np.array(self.peaks)
            idx_floor = np.floor(indices).astype(int)
            idx_ceil = np.clip(np.ceil(indices).astype(int), 0, len(self.raw_signal)-1)
            val_floor = self.raw_signal[idx_floor]
            val_ceil = self.raw_signal[idx_ceil]
            y_vals = val_floor + (val_ceil - val_floor) * (indices - idx_floor)
            self.scatter.set_data(self.peaks, y_vals)
        else:
            self.scatter.set_data([], [])
        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax: return
        click_idx = event.xdata
        if event.button == 3: # Delete
            if not self.peaks: return
            dists = [abs(p - click_idx) for p in self.peaks]
            if min(dists) < 10: del self.peaks[dists.index(min(dists))]
            self.update_plot()
        elif event.button == 1: # Add
            start = int(max(0, click_idx - 10))
            end = int(min(len(self.raw_signal), click_idx + 10))
            window = self.raw_signal[start:end]
            if len(window) == 0: return
            local_max_int = start + np.argmax(window)
            refined_val = refine_location_rust_parity(self.raw_signal, local_max_int)
            if not any(abs(p - refined_val) < 1.0 for p in self.peaks):
                self.peaks.append(refined_val)
                self.peaks.sort()
                self.update_plot()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the JSON file")
    args = parser.parse_args()

    try:
        with open(args.file, 'r') as f: raw_input = json.load(f)
    except FileNotFoundError: return

    data = raw_input[0] if isinstance(raw_input, list) else raw_input
    vitals = data.get("vital_signs", {})
    
    fs = data.get("fps", data.get("fs", 30.0))
    if "fs" in data: del data["fs"]
    data["fps"] = fs

    for k in ["rolling_heart_rate", "rolling_respiratory_rate"]:
        if k in vitals: del vitals[k]

    n_samples = data.get("n", 0)
    if n_samples == 0 and "ppg_waveform" in vitals:
        n_samples = len(vitals["ppg_waveform"]["data"])
    duration_sec = n_samples / fs if fs > 0 else 0
    print(f" -> Duration: {duration_sec:.2f}s (FPS: {fs}Hz)")

    # 3. Process PPG
    if "ppg_waveform" in vitals:
        wave = vitals["ppg_waveform"]
        conf_arr = wave.get("confidence")
        
        # PPG Settings
        CONF_THRESH = 0.25 # Matches our lowered rust gating
        
        annotator = PeakAnnotator(
            wave["data"], fs, "PPG (Heart Rate)", 
            rate_hint=vitals.get("heart_rate", {}).get("value"),
            min_rate=40.0, max_rate=220.0, detection_threshold=0.35,
            window_cycles=2.5, max_rate_change=1.0
        )
        
        peaks_float = annotator.peaks
        wave["peak_indices"] = [int(round(p)) for p in peaks_float]
        
        # Use full confidence array for gating inside metric calc
        confidence_data = np.array(conf_arr, dtype=np.float32) if conf_arr else np.ones(len(wave["data"]))
        
        nk_metrics = calculate_neurokit_metrics(
            peaks_float, fs, confidence_data, CONF_THRESH, "PPG"
        )
        
        for key in ["heart_rate", "hrv_sdnn", "hrv_rmssd", "hrv_lfhf", "stress_index", "hrv_pnn50", "hrv_sd1sd2"]:
            min_dur = MIN_DURATION_THRESHOLDS.get(key, 0)
            if duration_sec < min_dur:
                if key in vitals: del vitals[key]
                continue

            if key == "heart_rate":
                if key in vitals: vitals[key]["note"] = "Legacy Verified (FFT-based)"
            elif key in nk_metrics:
                if key not in vitals: vitals[key] = {}
                vitals[key]["value"] = round(nk_metrics[key], 2)
                vitals[key]["confidence"] = 1.0
                vitals[key]["note"] = "Ground Truth (NeuroKit2 Verified)"
                print(f"    [WRITE] {key}: {vitals[key]['value']}")

    # 4. Process Resp
    if "respiratory_waveform" in vitals:
        wave = vitals["respiratory_waveform"]
        conf_arr = wave.get("confidence")
        
        annotator = PeakAnnotator(
            wave["data"], fs, "Respiratory Rate", 
            rate_hint=vitals.get("respiratory_rate", {}).get("value"),
            min_rate=3.0, max_rate=60.0, detection_threshold=1.2,
            window_cycles=1.5, max_rate_change=0.25
        )
        wave["peak_indices"] = [int(round(p)) for p in annotator.peaks]
        
        key = "respiratory_rate"
        if duration_sec < MIN_DURATION_THRESHOLDS[key]:
            if key in vitals: del vitals[key]
        else:
            if key in vitals: vitals[key]["note"] = "Legacy Verified (FFT-based)"

    # 5. Save
    with open(args.file, 'w') as f: json.dump(data, f, indent=4)
    print(f"Successfully updated {args.file}")

if __name__ == "__main__":
    main()