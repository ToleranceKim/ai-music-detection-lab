"""
Audio Feature Extraction Module

AI 생성 음악 탐지를 위한 오디오 특징 추출 함수들
Sunday와 Afchar 논문에서 사용된 특징들 구현

주요 특징:
- MFCC: 음색 정보 (사람의 청각 시스템 모방)
- Spectral Features: 주파수 도메인 특성
- Temporal Features: 시간 도메인 특성
- Chroma: 음계 정보 (화성 구조)
"""

import numpy as np
import librosa
from typing import Optional, Tuple, Dict, List, Union
import warnings

def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512) -> Dict[str, np.ndarray]:
    """
    MFCC (Mel-frequency cepstral coefficients) 추출

    음성/음악의 가장 중요한 특징 중 하나로, 사람의 청각 시스템을 모방하여 음색(timbre) 정보를 효과적으로 표현합니다.
    AI 생성 음악은 종종 미묘하게 다른 MFCC 패턴을 보입니다.

    Args:
        audio: 오디오 신호 (1차원 numpy 배열)
        sr: 샘플레이트 (Hz)
        n_mfcc: MFCC 계수 개수 (default: 13)
        n_fft: FFT 윈도우 크기 (주파수 해상도 결정)
        hop_length: 프레임 간 이동 간격 (시간 해상도 결정)

    Returns:
        dict: MFCC 특징들
            - 'mfcc_mean': 각 계수의 시간축 평균 (shape: n_mfcc,)
            - 'mfcc_std': 각 계수의 시간축 표준편차 (shape: n_mfcc,)
            - 'mfcc_raw': 전체 MFCC 시퀀스 (shape: n_mfcc x time_frames)

    Example:
        >>> audio, sr = librosa.load('music.wav')
        >>> mfcc_features = extract_mfcc(audio, sr)
        >>> print(mfcc_features['mfcc_mean'].shape) # (13,)
    """

    # MFCC 계산
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    return {
        'mfcc_mean': np.mean(mfccs, axis=1),
        'mfcc_std': np.std(mfccs, axis=1),
        'mfcc_raw': mfccs
    }
