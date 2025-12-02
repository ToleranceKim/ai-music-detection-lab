"""
Audio I/O utilities.

AI 생성 음악 탐지를 위한 오디오 입출력 및 기본 처리 함수들.
다양한 오디오 포맷을 지원하며, 표준화된 형식으로 변환합니다. 
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple, List

def load_audio(file_path: str, sr: Optional[int] = None, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    오디오 파일을 로드하고 전처리합니다.

    다양한 오디오 포맷(MP3, WAV, FLAC 등)을 지원하며,
    AI 음악 탐지를 위한 표준화된 형식으로 변환합니다.

    Args:
        file_path: 오디오 파일 경로
        sr: 목표 샘플레이트 (None이면 원본 유지,
            AI 탐지용으로는 16000 또는 22050 권장)
        mono: True면 스테레오를 모노로 변환 (탐지 모델용)

    Returns:
        tuple: (오디오 numpy 배열, 실제 샘플레이트)
            - audio: 정규화된 float32 배열 [-1, 1]
            - sr: 샘플레이트 (Hz)

    Raises:
        IOError: 파일 로드 실패 시

    Examples:
        >>> audio, sr = load_audio("music.mp3", sr=16000)
        >>> print(f"Duration: {len(audio)/sr:.2f} seconds")
        Duration: 180.25 seconds

    Note:
        - librosa 내부적으로 ffmpeg 사용하여 다양한 포맷 지원
        - 기본적으로 [-1, 1] 범위로 정규화됨
    """
    try:
        # librosa.load가 알아서 mp3, wav 등을 읽어줌
        audio, sampling_rate = librosa.load(file_path, sr=sr, mono=mono)
        return audio, sampling_rate
    except Exception as e:
        raise IOError(f"오디오 파일을 읽을 수 없습니다: {file_path}\n에러: {e}")


def save_audio(audio: np.ndarray, sr: int, file_path: str, format: Optional[str] = None) -> None:
    """
    오디오 데이터를 파일로 저장합니다.

    처리된 오디오나 생성된 오디오를 다양한 포맷으로 저장합니다.
    AI 탐지 결과 검증용 샘플 저장에 사용됩니다.

    Args:
        audio: 저장할 오디오 배열 ([-1, 1] 범위 권장)
        sr: 샘플레이트 (Hz)
        file_path: 저장할 파일 경로 (확장자로 포맷 자동 감지)
        format: 파일 포맷 지정 (None이면 확장자로 자동 감지)
                'WAV', 'FLAC', 'OGG' 등

    Raises:
        IOError: 파일 저장 실패 시

    Example:
        >>> save_audio(processed_audio, 16000, "output.wav")
        오디오 저장 완료: output.wav

    Note:
        - WAV: 무손실, 호환성 최고
        - FLAC: 무손실 압축, 파일 크기 작음
        - OGG: 손실 압축, 웹 호환성 좋음
    """

    try:
        # soundfile.write가 자동으로 형식 감지
        sf.write(file_path, audio, sr, format=format)
        print(f"오디오 저장 완료: {file_path}")
    except Exception as e:
        raise IOError(f"오디오 저장 실패: {file_path}\n에러: {e}")


def split_audio(audio: np.ndarray, sr: int, segment_duration: float = 10.0) -> List[np.ndarray]:
    f"""
    긴 오디오를 고정 길이 세그먼트로 분할합니다.

    AI 모델 입력을 위해 긴 오디오를 일정한 길이로 자릅니다.
    Sunday 논문은 10초, Afchar 논문은 3초 세그먼트를 사용합니다.

    Args:
        audio: 입력 오디오 배열
        sr: 샘플레이트 (Hz)
        segment_duration: 세그먼트 길이 (초, 기본 10초)

    Returns:
        List[np.ndarray]: 분할된 세그먼트 리스트
            - 각 세그먼트는 동일한 길이
            - 마지막 세그먼트는 제로 패딩됨

    Example:
        >>> segments = split_audio(long_audio, 16000, 10.0)
        >>> print(f"Created {len(segments)} segments")
        Created 18 segments

    Note:
        - 마지막 세그먼트가 짧으면 0으로 패딩
        - overlap 옵션 추가 고려 (데이터 증강용)
        - 세그먼트 길이는 모델 아키텍처에 따라 결정
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
    f"""
    오디오 신호를 정규화합니다.

    일관된 볼륨 레벨로 오디오를 정규화하여 모델 성능을 향상시킵니다.

    Args:
        audio: 입력 오디오 배열
        method: 정규화 방법
            - 'peak': 최대값 기준 (전체 다이나믹 레인지 사용)
            - 'rms': RMS 기준 (평균 음량 일치)

    Returns:
        np.ndarray: 정규화된 오디오
            - peak: [-1, 1] 범위
            - rms: 목표 RMS 레벨 (0.1)

    Example:
        >>> normalized = normalize_audio(audio, 'peak')
        >>> print(f"Max value: {np.max(np.abs(normalized)):.3f}")
        Max value: 1.000

    Note:
        - peak: 클리핑 방지, 전체 범위 활용
        - rms: 지각적 음량 일치, 더 자연스러운 소리
        - AI 탐지에는 peak 방법이 더 일반적
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
