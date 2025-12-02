"""
Audio I/O utilities.
Handles loading, saving, and basic audio processing.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple, List

def load_audio(file_path: str, sr: Optional[int] = None, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    오디오 파일을 불러오는 함수

    예시 : 
        audio, sr = load_audio("music.mp3", sr=16000)
    """
    try:
        # librosa.load가 알아서 mp3, wav 등을 읽어줌
        audio, sampling_rate = librosa.load(file_path, sr=sr, mono=mono)
        return audio, sampling_rate
    except Exception as e:
        raise IOError(f"오디오 파일을 읽을 수 없습니다: {file_path}\n에러: {e}")

def save_audio(audio: np.ndarray, sr: int, file_path: str, format: Optional[str] = None) -> None:
    """
    오디오 파일을 저장하는 함수

    예시:
        save_audio(audio_data, 16000, "output.wav")
    """
    try:
        # soundfile.write가 자동으로 형식 감지
        sf.write(file_path, audio, sr, format=format)
        print(f"오디오 저장 완료: {file_path}")
    except Exception as e:
        raise IOError(f"오디오 저장 실패: {file_path}\n에러: {e}")

def split_audio(audio: np.ndarray, sr: int, segment_duration: float = 10.0) -> List[np.ndarray]:
    """
    긴 오디오를 10초씩 잘라내는 함수

    예시:
        segments = split_audio(long_audio, 16000, 10.0)
        # 결과: [10초 오디오1, 10초 오디오2, ...]
    """
    # 10초가 몇 개의 샘플인지 계산
    segment_length = int(segment_duration * sr)
    
    # 전체 오디오를 segment_length씩 잘라내기
    segments = []
    for start in range(0, len(audio), segment_length):
        end = start + segment_length
        segment = audio[start:end]

        # 마지막 부분이 10초보다 짧으면 제로 패딩
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))

        segments.append(segment)
    
    return segments

def normalize_audio(audio: np.ndarray, method: str = 'peak') -> np.ndarray:
    """
    오디오 볼륨을 정규화하는 함수

    예시:
        normalized = normalize_audio(audio, 'peak')
        # 결과: -1 ~ 1 사이의 값으로 정규화된 오디오
    """
    if method == 'peak':
        # 가장 큰 값을 1로 만들기
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    elif method == 'rms':
        # RMS(평균 볼륨) 기준 정규화
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            return audio / rms * 0.1 # 0.1은 목표 RMS 값
        return audio

    else:
        raise ValueError(f"Unknown normalization method: {method}")
