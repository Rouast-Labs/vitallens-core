import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import vitallens_core
from scipy.signal import butter, filtfilt

VIRTUAL_FS = 1000.0
# Interval limits follow the 40-220 bpm heart rate range, as in the core
MIN_RR_MS = 60000.0 / 220.0
MAX_RR_MS = 60000.0 / 40.0
MAX_REL_CHANGE = 0.25
# The core low-passes the PPG before HRV peak detection at max(3 Hz, 1.3 x HR)
HRV_LOWPASS_HZ = 3.0
HRV_LOWPASS_RATE_FACTOR = 1.3

MIN_DURATION_THRESHOLDS = {
    "heart_rate": 5.0,
    "respiratory_rate": 10.0,
    "hrv_sdnn": 20.0,
    "hrv_rmssd": 20.0,
    "hrv_pnn50": 30.0,
    "hrv_sd1sd2": 120.0,
    "hrv_lfhf": 120.0,
    "stress_index": 55.0
}

def lowpass_rust_parity(signal, fs, rate_hint):
    """2nd-order Butterworth, forward-backward, matching the core's HRV low-pass."""
    cutoff = HRV_LOWPASS_HZ if rate_hint is None else max(HRV_LOWPASS_HZ, HRV_LOWPASS_RATE_FACTOR * rate_hint / 60.0)
    signal = np.asarray(signal, dtype=np.float64)
    if cutoff >= fs / 2 or len(signal) <= 9:
        return signal
    return filtfilt(*butter(2, cutoff / (fs / 2)), signal)

def refine_location_rust_parity(signal, idx):
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
    intervals = []
    used_confs = []
    
    for i in range(len(peaks_indices) - 1):
        p1_idx_float = peaks_indices[i]
        p2_idx_float = peaks_indices[i+1]
        
        idx1 = int(round(p1_idx_float))
        idx2 = int(round(p2_idx_float))
        
        if idx1 >= len(confidence_array) or idx2 >= len(confidence_array):
            continue
            
        c1 = confidence_array[idx1]
        c2 = confidence_array[idx2]
        
        if c1 >= conf_threshold and c2 >= conf_threshold:
            t1 = p1_idx_float / fs
            t2 = p2_idx_float / fs
            diff_ms = (t2 - t1) * 1000.0

            if diff_ms > 0:
                intervals.append((diff_ms, t2))
                used_confs.extend([float(c1), float(c2)])

    min_conf_peaks = min(used_confs) if used_confs else 0.0
    return intervals, min_conf_peaks

def outlier_mask_rust_parity(intervals_ms):
    intervals_ms = [float(x) for x in intervals_ms]
    if len(intervals_ms) < 2: return [True] * len(intervals_ms)

    valid_range = [x for x in intervals_ms if MIN_RR_MS <= x <= MAX_RR_MS]
    if not valid_range: return [False] * len(intervals_ms)

    valid_range.sort()
    mid_idx = len(valid_range) // 2
    median = valid_range[mid_idx]

    keep = []
    last_accepted = median

    for curr in intervals_ms:
        ok = MIN_RR_MS <= curr <= MAX_RR_MS and abs(curr - last_accepted) <= last_accepted * MAX_REL_CHANGE
        keep.append(ok)
        if ok: last_accepted = curr

    return keep

def calculate_neurokit_metrics(peaks_indices_float, fs, confidence_array, conf_threshold, label):
    if len(peaks_indices_float) < 2: return {}, 0.0

    raw_intervals, min_conf_peaks = extract_intervals_rust_parity(peaks_indices_float, fs, confidence_array, conf_threshold)
    keep = outlier_mask_rust_parity([iv for iv, _ in raw_intervals])
    clean = [x for x, k in zip(raw_intervals, keep) if k]

    if len(clean) < 2: return {}, 0.0

    # Intervals with the time of their ending beat: NeuroKit2 then only pairs truly consecutive
    # intervals for successive differences, and places them correctly on the LF/HF time axis
    rri = {"RRI": np.array([iv for iv, _ in clean]), "RRI_Time": np.array([t for _, t in clean])}
    metrics = {}

    try:
        td = nk.hrv_time(rri, sampling_rate=VIRTUAL_FS)
        metrics['hrv_sdnn'] = td['HRV_SDNN'].values[0]
        metrics['hrv_rmssd'] = td['HRV_RMSSD'].values[0]
        if 'HRV_pNN50' in td: metrics['hrv_pnn50'] = td['HRV_pNN50'].values[0]
    except Exception as e:
        print(f"[{label}] Time domain error: {e}")

    try:
        fd = nk.hrv_frequency(rri, sampling_rate=VIRTUAL_FS, psd_method='fft')
        metrics['hrv_lfhf'] = fd['HRV_LFHF'].values[0]
    except Exception as e:
        print(f"[{label}] LF/HF Error: {e}")

    try:
        nl = nk.hrv_nonlinear(rri, sampling_rate=VIRTUAL_FS)
        # NeuroKit2's HRV_SI is the slope index, not Baevsky's stress index, so it is not used
        if 'HRV_SD1SD2' in nl: metrics['hrv_sd1sd2'] = nl['HRV_SD1SD2'].values[0]
    except Exception as e:
        print(f"[{label}] Nonlinear error: {e}")

    avg_conf = float(np.mean(confidence_array)) if len(confidence_array) > 0 else 0.0
    final_conf = min(avg_conf, min_conf_peaks)

    return metrics, final_conf

class PeakAnnotator:
    def __init__(self, raw_signal, fs, title, rate_hint=None,
                 min_rate=40.0, max_rate=220.0, detection_threshold=0.45,
                 window_cycles=2.5, max_rate_change=1.0, interactive=True, existing_peaks=None):
        self.raw_signal = np.array(raw_signal)
        self.fs = fs
        self.title = title
        
        print(f"[{title}] Auto-detecting peaks (Rust Parity)...")
        
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

        if existing_peaks:
            self.peaks = self.snap_to_detections(existing_peaks, self.peaks)
            print(f"[{title}] Starting from {len(self.peaks)} existing annotations.")

        if not interactive:
            return

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.t = np.arange(len(self.raw_signal))
        self.ax.plot(self.t, self.raw_signal, color='#444444', alpha=0.8, label='Signal')
        self.scatter, = self.ax.plot([], [], 'rx', markersize=12, markeredgewidth=2, label='Peaks')
        self.ax.set_title(f"{title}\nL-Click: Add | R-Click: Delete | Close: Save")
        self.ax.legend(loc='upper left')
        self.update_plot()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def snap_to_detections(self, annotated, detected, tolerance_s=0.15):
        """Keeps the annotated beats, taking sub-sample positions from the nearest detection or local refinement."""
        detected = np.asarray(detected)
        snapped = []
        for a in annotated:
            if len(detected) and np.min(np.abs(detected - a)) <= tolerance_s * self.fs:
                snapped.append(float(detected[np.argmin(np.abs(detected - a))]))
            else:
                lo, hi = max(0, int(a) - 2), min(len(self.raw_signal), int(a) + 3)
                snapped.append(refine_location_rust_parity(self.raw_signal, lo + int(np.argmax(self.raw_signal[lo:hi]))))
        return sorted(snapped)

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
        if event.button == 3:
            if not self.peaks: return
            dists = [abs(p - click_idx) for p in self.peaks]
            if min(dists) < 10: del self.peaks[dists.index(min(dists))]
            self.update_plot()
        elif event.button == 1:
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
    parser.add_argument("--no-gui", action="store_true", help="Accept the auto-detected peaks without annotation")
    args = parser.parse_args()

    try:
        with open(args.file, 'r') as f: raw_input = json.load(f)
    except FileNotFoundError: return

    data = raw_input[0] if isinstance(raw_input, list) else raw_input
    
    waveforms = data.setdefault("waveforms", {})
    vitals = data.setdefault("vitals", {})
        
    fs = data.get("fps", data.get("fs", 30.0))
    if "fs" in data: del data["fs"]
    data["fps"] = fs

    if "rolling_vitals" in data:
        data["rolling_vitals"] = {}

    n_samples = data.get("n", 0)
    if n_samples == 0 and "ppg_waveform" in waveforms:
        n_samples = len(waveforms["ppg_waveform"]["data"])
    duration_sec = n_samples / fs if fs > 0 else 0
    print(f" -> Duration: {duration_sec:.2f}s (FPS: {fs}Hz)")

    if "ppg_waveform" in waveforms:
        wave = waveforms["ppg_waveform"]
        conf_arr = wave.get("confidence")
        
        CONF_THRESH = 0.5
        hr_hint = vitals.get("heart_rate", {}).get("value")

        # Annotate on the low-passed waveform so peak positions match what the core uses for HRV
        annotator = PeakAnnotator(
            lowpass_rust_parity(wave["data"], fs, hr_hint), fs, "PPG (Heart Rate)",
            rate_hint=hr_hint,
            min_rate=40.0, max_rate=220.0, detection_threshold=0.35,
            window_cycles=2.5, max_rate_change=1.0, interactive=not args.no_gui,
            existing_peaks=wave.get("peak_indices")
        )
        
        peaks_float = annotator.peaks
        # Without the GUI, existing annotations are kept as they are
        if not (args.no_gui and wave.get("peak_indices")):
            wave["peak_indices"] = [int(round(p)) for p in peaks_float]
        
        confidence_data = np.array(conf_arr, dtype=np.float32) if conf_arr else np.ones(len(wave["data"]))
        
        nk_metrics, nk_conf = calculate_neurokit_metrics(
            peaks_float, fs, confidence_data, CONF_THRESH, "PPG"
        )
        
        for key in ["heart_rate", "hrv_sdnn", "hrv_rmssd", "hrv_lfhf", "hrv_pnn50", "hrv_sd1sd2"]:
            min_dur = MIN_DURATION_THRESHOLDS.get(key, 0)
            if duration_sec < min_dur:
                if key in vitals: del vitals[key]
                continue

            if key == "heart_rate":
                if key in vitals: vitals[key]["note"] = "Legacy Verified (FFT-based)"
            elif key in nk_metrics:
                if key not in vitals: vitals[key] = {}
                vitals[key]["value"] = round(nk_metrics[key], 2)
                vitals[key]["confidence"] = round(float(nk_conf), 4)
                vitals[key]["note"] = "Ground Truth (NeuroKit2 Verified)"
                print(f"    [WRITE] {key}: {vitals[key]['value']} (conf: {vitals[key]['confidence']})")

    if "respiratory_waveform" in waveforms:
        wave = waveforms["respiratory_waveform"]
        conf_arr = wave.get("confidence")
        
        annotator = PeakAnnotator(
            wave["data"], fs, "Respiratory Rate", 
            rate_hint=vitals.get("respiratory_rate", {}).get("value"),
            min_rate=3.0, max_rate=60.0, detection_threshold=1.2,
            window_cycles=1.5, max_rate_change=0.25, interactive=not args.no_gui,
            existing_peaks=wave.get("peak_indices")
        )
        if not (args.no_gui and wave.get("peak_indices")):
            wave["peak_indices"] = [int(round(p)) for p in annotator.peaks]
        
        key = "respiratory_rate"
        if duration_sec < MIN_DURATION_THRESHOLDS[key]:
            if key in vitals: del vitals[key]
        else:
            if key in vitals: vitals[key]["note"] = "Legacy Verified (FFT-based)"

    with open(args.file, 'w') as f: json.dump(data, f, indent=4)
    print(f"Successfully updated {args.file}")

if __name__ == "__main__":
    main()