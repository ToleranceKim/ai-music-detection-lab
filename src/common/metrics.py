"""
AI 음악 탐지 모델을 위한 평가 메트릭.

이진 분류 모델 평가를 위한 다양한 메트릭들을 제공하며,
특히 AI 생성 음악 탐지에 최적화되어 있습니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    classification_report, roc_auc_score,
    precision_recall_fscore_support
)
from typing import List, Tuple, Optional, Union, Dict
import warnings
warnings.filterwarnings('ignore')

# Seaborn 스타일 설정
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

def calculate_accuracy(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    """
    이진 분류의 정확도를 계산합니다.

    Args:
        predictions: 모델 예측값 (확률 또는 로짓)
        labels: 실제 레이블 (0 또는 1)
        threshold: 이진 분류를 위한 결정 경계값

    Returns:
        0과 1 사이의 정확도 점수

    Examples:
        >>> preds = np.array([0.7, 0.3, 0.8, 0.2])
        >>> labels = np.array([1, 0, 1, 0])
        >>> acc = calculate_accuracy(preds, labels)
        >>> print(f"Accuracy: {acc:.4f}")
        Accuracy: 1.0000
    """
    # 확률값을 이진 예측으로 변환
    if predictions.ndim > 1:
        predictions = predictions[:, 1] # positive class 확률

    binary_preds = (predictions >= threshold).astype(int)

    # 정확도 계산
    correct = (binary_preds == labels).sum()
    total = len(labels)

    return correct / total if total > 0 else 0.0


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    이진 분류를 위한 종합적인 메트릭을 계산합니다.

    Args:
        predictions: 모델 예측값
        labels: 실제 레이블
        threshold: 결정 경계값

    Returns:
        다양한 메트릭을 포함한 딕셔너리

    Note:
        Class 0 = Human (정상 음악)
        Class 1 = Deepfake (AI 생성 음악)
    """
    # 이진 예측으로 변환
    if predictions.ndim > 1:
        predictions = predictions[:, 1]

    binary_preds = (predictions >= threshold).astype(int)

    # Confusion Matrix 요소들
    tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()

    # 각종 메트릭 계산
    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0, # TPR
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0, # TNR
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0, # False Positive Rate
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0, # False Negative Rate
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

    return metrics

# AUC-ROC

def calculate_auc_roc(predictions: np.ndarray, labels: np.ndarray, return_curve: bool = False) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """
    AUC-ROC 점수를 계산하고 선택적으로 ROC 곡선을 반환합니다.

    Args:
        predictions: 모델 예측값 (확률)
        labels: 실제 레이블
        return_curve: FPR과 TPR 배열 반환 여부

    Returns:
        AUC 점수, 또는 return_curve=True일 때 (AUC, FPR, TPR) 튜플

    Examples:
        >>> auc_score = calculate_auc_roc(preds, labels)
        >>> auc_score, fpr, tpr = calculate_auc_roc(preds, labels, return_curve=True)
    """
    if predictions.ndim > 1:
        predictions = predictions[:, 1]

    # ROC curve 계산
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    auc_score = auc(fpr, tpr)

    if return_curve:
        return auc_score, fpr, tpr
    return auc_score

# EER

def calculate_eer(predictions: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Equal Error Rate (EER) - FPR과 FNR이 같아지는 지점을 계산합니다.

    Args:
        predictions: 모델 예측값
        labels: 실제 레이블

    Returns:
        (EER 값, EER 임계값) 튜플

    Note:
        EER은 보안 시스템에서 중요한 메트릭으로,
        사람 음악을 AI로 오인하는 비율과
        AI 음악을 사람으로 놓치는 비율이 같아지는 지점입니다.
    """
    if predictions.ndim > 1:
        predictions = predictions[:, 1]

    # ROC curve 계산
    fpr, tpr, thresholds = roc_curve(labels, predictions)

    # FNR = 1 - TPR
    fnr = 1 - tpr

    # FPR과 FNR의 차이가 최소인 지점 찾기
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]

    return eer, eer_threshold

    
# 시각화

def plot_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5,
 normalize: bool = True, title: Optional[str] = None, save_path: Optional[str] = None) -> None:
    """
    혼동 행렬을 시각화합니다.

    Args:
        predictions: 모델 예측값
        labels: 실제 레이블
        threshold: 결정 경계값
        normalize: 백분율 표시 여부
        title: 플롯 제목
        save_path: 저장 경로
    """
    # 이진 예측으로 변환
    if predictions.ndim > 1:
        predictions = predictions[:, 1]

    binary_preds = (predictions >= threshold).astype(int)

    # Confusion matrix 계산
    cm = confusion_matrix(labels, binary_preds)

    # 정규화
    if normalize:
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 플롯 생성
    fig, ax = plt.subplots(figsize=(8, 6))

    # Heatmap 그리기
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', square=True, cbar=True, ax=ax)

    # 각 셀에 값과 퍼센트 표시
    for i in range(2):
        for j in range(2):
            if normalize:
                text = f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)'
            else:
                text = f'{cm[i, j]}'

            color = 'white' if cm[i,j] > cm.max() / 2 else 'black'
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', color=color, fontsize=14)

    # 라벨 설정
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xticklabels(['Human (0)', 'Deepfake (1)'])
    ax.set_yticklabels(['Human (0)', 'Deepfake (1)'])

    # 타이틀 설정
    if title is None:
        title = 'Confusion Matrix'
    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()

    # 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"혼동 행렬이 {save_path}에 저장되었습니다")

    plt.show()


def plot_roc_curve(predictions: np.ndarray, labels: np.ndarray, title: Optional[str] = None, save_path: Optional[str] = None) -> None:
    """
    ROC 곡선과 AUC 점수를 시각화합니다.

    Args:
        predictions: 모델 예측값
        labels: 실제 레이블
        title: 플롯 제목
        save_path: 저장 경로
    """
    if predictions.ndim > 1:
        predictions = predictions[:, 1]

    # ROC curve 계산
    fpr, tpr, _ = roc_curve(labels, predictions)
    auc_score = auc(fpr, tpr)

    # EER 계산
    eer, eer_threshold = calculate_eer(predictions, labels)

    # 플롯 생성
    plt.figure(figsize=(8, 6))

    # ROC curve
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')

    # 대각선 (랜덤 분류기)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')

    # EER 지점 표시
    eer_index = np.argmin(np.abs(fpr - eer))
    plt.plot(fpr[eer_index], tpr[eer_index], 'go', markersize=10, label=f'EER = {eer:.3f}')

    # 라벨과 타이틀
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)

    if title is None:
        title = 'ROC Curve'
    plt.title(title, fontsize=14, pad=20)

    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC 곡선이 {save_path}에 저장되었습니다.")

    plt.show()


def generate_evaluation_report(predictions: np.ndarray, labels: np.ndarray, model_name: str = "모델") -> str:
    """
    종합적인 평가 리포트를 생성합니다.

    Args:
        predictions: 모델 예측값
        labels: 실제 레이블
        model_name: 모델 이름

    Returns:
        포맷팅된 리포트 문자열
    """
    # 메트릭 계산
    metrics = calculate_metrics(predictions, labels)
    auc_score = calculate_auc_roc(predictions, labels)
    eer, eer_threshold = calculate_eer(predictions, labels)

    # 리포트 생성
    report = f"""
    {'='*60}
    {model_name} 평가 리포트
    {'='*60}

    분류 메트릭:
    ----------
    정확도 (Accuracy):    {metrics['accuracy']:.4f}
    정밀도 (Precision):   {metrics['precision']:.4f}
    재현율 (Recall):      {metrics['recall']:.4f}
    F1-Score:            {metrics['f1_score']:.4f}
    특이도 (Specificity): {metrics['specificity']:.4f}

    오류율:
    ----------
    오탐율 (FPR): {metrics['fpr']:.4f}
    미탐율 (FNR): {metrics['fnr']:.4f}
    EER:         {eer:.4f} (임계값: {eer_threshold:.4f})

    ROC 분석:
    ----------
    AUC-ROC:    {auc_score:.4f}

    혼동 행렬:
    ----------
    True Positives (AI를 AI로):     {metrics['tp']:.0f}
    True Negatives (사람을 사람으로):  {metrics['tn']:.0f}
    False Positives (사람을 AI로):   {metrics['fp']:.0f}
    False Negatives (AI를 사람으로):  {metrics['fn']:.0f}

    {'='*60}
    """

    return report

# 테스트용 메인 함수
if __name__ == "__main__":
    # 테스트용 더미 데이터
    np.random.seed(42)
    n_samples = 1000

    # 가상의 예측값과 레이블 생성
    labels = np.random.randint(0, 2, n_samples)
    predictions = np.random.rand(n_samples)

    # 메트릭 계산 및 출력
    print(generate_evaluation_report(predictions, labels, "Test Model"))

    # 시각화
    plot_confusion_matrix(predictions, labels, title="Test Model - Confusion Matrix")
    plot_roc_curve(predictions, labels, title="Test Model - ROC Curve")

