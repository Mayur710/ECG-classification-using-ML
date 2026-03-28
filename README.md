# Evolutionary ECG Classification Pipeline

A full end-to-end machine learning pipeline for cardiac arrhythmia detection, built on the MIT-BIH Arrhythmia Database. This project documents the complete journey from raw signal processing through classical ML to deep learning — including a documented GA failure that revealed a fundamental limitation of hand-crafted features.

---

## Data Source

**MIT-BIH Arrhythmia Database**
- 48 patient records sampled at 360 Hz
- Two-channel ambulatory ECG recordings, each approximately 30 minutes long
- Expert cardiologist annotations for every heartbeat (~110,000 total beats)
- Source: [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)

Beat classes follow the AAMI EC57 standard:

| Class | Meaning | Count (approx) |
|-------|---------|----------------|
| N | Normal | ~25,000 |
| S | Supraventricular | ~1,500 |
| V | Ventricular | ~3,000 |
| F | Fusion | ~300 |
| Q | Unknown/Paced | ~200 |

---

## Pipeline Overview

```
Raw ECG Signal (.dat)
        ↓
Phase 1: Signal Preprocessing (Butterworth DSP Filter)
        ↓
Phase 2: Feature Engineering (R-peak detection + CSV assembly)
        ↓
Phase 3: Genetic Algorithm Feature Selection ← FAILED (documented below)
        ↓
Phase 4: Classical ML (MLP → Random Forest Binary Classifier)
        ↓
Phase 5: Deep Learning (1D CNN on raw waveforms) 
```

---

## Phase 1 — Signal Preprocessing

**Problem:** Raw ECG signals contain two types of noise:
- Baseline wander (< 0.5 Hz) caused by patient breathing
- Powerline interference (50/60 Hz) from electrical equipment

**Solution:** A 5th-order Butterworth Bandpass Filter (0.5–50 Hz) implemented using `scipy.signal`.

```python
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    b, a    = butter(order, [lowcut/nyquist, highcut/nyquist], btype='band')
    return filtfilt(b, a, data)
```

`filtfilt` applies the filter twice (forward + backward) to eliminate phase distortion, preserving the precise timing of R-peaks.

**Output:** 48 cleaned `.npy` files saved to `processed_data/`

---

## Phase 2 — Feature Engineering & Labeling

**R-peak detection** using `scipy.signal.find_peaks` with a minimum distance of 100 samples (0.28s) and a height threshold of 50% of the signal maximum — approximating the Pan-Tompkins algorithm logic.

**Features extracted per heartbeat:**

| Feature | Description |
|---------|-------------|
| RR_Interval | Time between consecutive beats (seconds) |
| Heart_Rate | Instantaneous BPM |
| QRS_Width | Duration of ventricular contraction (ms) |
| R_Amplitude | Peak voltage of the heartbeat (mV) |

**Ground truth labeling** was performed by matching detected R-peaks against the official `.atr` annotation files within a ±36 sample (0.1 second) tolerance window. Unmatched peaks were discarded as false detections.

Labels were then remapped to AAMI standard classes (e.g., L, R, e → N; V, E → V).

**Output:** `Final_ECG_Dataset.csv` containing ~30,000 labeled heartbeats from all 48 patients.

---

## Phase 3 — Genetic Algorithm Feature Selection

**Objective:** Find the optimal subset of features that maximises diagnostic accuracy.

**Implementation:**
- Chromosomes: binary strings where `1` = keep feature, `0` = drop
- Selection: top 50% survive each generation (tournament-style)
- Crossover: single-point crossover between two parents
- Mutation: bit-flip at 10% probability per gene
- Fitness function: 3-fold cross-validated accuracy on a Decision Tree

**Result:** After 15 generations, the GA converged on a single feature — `RR_Interval` — achieving 85.8% accuracy.

### Why the GA Failed

This looked like success. It was not.

The dataset contains ~84% Normal beats. The GA optimised for **global accuracy**, not clinical sensitivity. It discovered that predicting "Normal" for almost every beat yields ~84% accuracy without detecting a single arrhythmia. To verify this, a dummy classifier that always predicts Normal scores ~84% — nearly identical to the GA result.

The GA correctly dropped QRS_Width and R_Amplitude because they added no value to the accuracy metric it was optimising. In doing so, it eliminated the only features that distinguish arrhythmia morphology.

**Root cause:** The fitness function rewarded the wrong objective. The correct metric for imbalanced medical data is `balanced_accuracy` or `f1_macro`, not raw accuracy.

This failure mathematically proved that timing features alone cannot distinguish arrhythmia classes — a finding that motivated the entire Phase 4 redesign.

---

## Phase 4 — Model Evolution

### Attempt 1: MLP with SMOTE
Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance all 5 classes, then trained a Multi-Layer Perceptron (128×64 architecture).

**Result:** ~50% accuracy. SMOTE balanced the classes, but `RR_Interval` and `Heart_Rate` are mathematically indistinguishable across arrhythmia types. No amount of balancing fixes an insufficient feature set.

### Attempt 2: Binary Random Forest (Final Classical Model)
Pivoted to binary classification (Normal vs. Arrhythmia) using a Random Forest with `class_weight='balanced'` on all 4 features.

**Final results:**

| Metric | Score |
|--------|-------|
| Accuracy | ~61% |
| AUC (ROC) | ~0.70 |

This represents the **performance ceiling of hand-crafted features**. The model is genuinely learning signal — 0.70 AUC is well above random guessing (0.50) — but cannot exceed this ceiling without access to raw waveform shape.

---

## Phase 5 — 1D CNN 

The classical pipeline proved that extracted timing and morphology features have a hard performance ceiling. The correct solution is to bypass feature extraction entirely and let a Convolutional Neural Network learn its own features directly from raw heartbeat waveforms.

**Planned architecture:**
- Input: segmented heartbeat windows (360 samples per beat, centred on R-peak)
- Conv1D layers to detect local waveform patterns (P-wave, QRS complex, T-wave)
- MaxPooling for translation invariance
- Dense layers for final classification

 90%+ accuracy based on published literature using identical data.

---

## Project Structure

```
Evolutionary_ECG_Project/
├── data/                        # Raw MIT-BIH .dat, .hea, .atr files
├── processed_data/              # Cleaned signals (48 x _clean.npy)
├── features/                    # Per-patient feature CSVs
├── Final_ECG_Dataset.csv        # Master labeled dataset (~30,000 beats)
├── final_binary_ecg_model.joblib  # Saved Random Forest model
└── notebooks/
    ├── ecg_pipeline.ipynb       # Main pipeline (Phases 1–4)
    └── cnn_phase.ipynb          # Phase 5 
```

---

## Dependencies

```
wfdb
numpy
scipy
pandas
scikit-learn
imbalanced-learn
matplotlib
joblib
tensorflow / keras  (Phase 5)
```

Install all:
```bash
pip install wfdb numpy scipy pandas scikit-learn imbalanced-learn matplotlib joblib tensorflow
```

---

## Key Findings

1. **DSP filtering is non-negotiable.** Unfiltered signals contain enough noise to corrupt peak detection entirely.
2. **The GA optimised the wrong objective.** Global accuracy on imbalanced data is a misleading fitness function for any medical classification task.
3. **Timing features have a hard ceiling.** RR intervals and heart rate cannot distinguish arrhythmia morphology — two beats with identical timing can have completely different shapes and diagnoses.
4. **Binary classification is more honest than multi-class on insufficient features.** The pivot from 5-class to binary improved clinical usefulness even at lower feature richness.
5. **The performance ceiling is a feature, not a bug.** It provides a clean, quantified justification for transitioning to deep learning.

---

## References

- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology* 20(3):45-50 (May-June 2001).
- Pan J, Tompkins WJ. A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering* 32(3):230-236 (1985).
- AAMI EC57:1998 — Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms.
