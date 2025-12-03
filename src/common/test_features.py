"""
Feature extraction 테스트
"""

import numpy as np
from src.common.features import *

def test_features():
    print("="*50)
    print("Feature Extraction Tests")
    print("="*50)

    # 테스트 오디오 데이터
    sr = 16000
    duration = 2
    t = np.linspace(0, duration, sr * duration)
    audio = np.sin(2 * np.pi * 440 * t) + 1.0 * np.random.randn(len(t))

    print("\n[1] Individual features test...")

    # 각 특징 테스트
    mfcc = extract_mfcc(audio, sr)
    print(f" MFCC shape: {mfcc['mfcc_mean'].shape}")

    centroid = extract_spectral_centroid(audio, sr)
    print(f"Centroid: {centroid['spectral_centroid_mean']:.2f} Hz")

    rolloff = extract_spectral_rolloff(audio, sr)
    print(f"Rolloff: {rolloff['spectral_rolloff_mean']:.2f} Hz")

    contrast = extract_spectral_contrast(audio, sr)
    print(f"Contrast bands: {len(contrast['spectral_contrast_mean'])}")

    zcr = extract_zero_crossing_rate(audio)
    print(f"ZCR: {zcr['zcr_mean']:.4f}")

    rms = extract_rms_energy(audio)
    print(f"RMS: {rms['rms_mean']:.4f}")

    chroma = extract_chroma(audio, sr)
    print(f"Chroma bins: {len(chroma['chroma_mean'])}")

    print("\n[2] All feature test...")
    all_features = extract_all_features(audio, sr)
    print(f"Total features: {len(all_features)}")

    print("\n All tests passed!")

if __name__ == "__main__":
    test_features()
