# Detecting Musical Deepfakes

- title: "[논문 리뷰] Detecting Musical Deepfakes – FakeMusicCaps와 ResNet18 기반 AI 음악 딥페이크 탐지"
- date: 2025-12-05
- tags: [AI Music Detection, Deepfake, FakeMusicCaps, ResNet18]
---

# Detecting Musical Deepfakes: FakeMusicCaps와 ResNet18으로 살펴본 음악 딥페이크 탐지

## 리뷰를 시작하며

첫 리뷰에서는 AI Music Detection 쪽에서 구현과 실험 구성이 비교적 직관적인 논문을 함께 살펴보려 합니다. 복잡한 모델과 방어 기법으로 바로 들어가기보다, 재현 가능한 예제를 통해 "생성 음악을 기계가 어떻게 구분하지"를 먼저 함께 체감해 본느 것이 목표입니다.

오늘 다룰 Nicholas Sunday의 「Detecting Musical Deepfakes」는 FakeMusicCaps 데이터셋의 오디오를 Mel Spectrogram으로 변환한 뒤 ResNet18 이진 분류기를 학습하는 기본적인 탐지 파이프라인을 제안합니다. 논문과 함꼐 공식 GitHub 레포지토리가 공개되어 있어 코드를 내려받아 곧바로 학습과 평가를 재현해 볼 수 있고, 전처리와 실험 설정을 바꾸어 보는 것도 비교적 수월합니다.

저자는 Deezer의 연구(Afchar et al, 2024)와 SONICS, FakeMusicCaps 논문을 주요 관련 연구로 검토합니다. 특히 Deezer 연구에서 사용된 피치 쉬프트와 템포 스트레치 조작 시나리오를 FakeMusicCaps와 ResNet18 구성에 다시 적용해, 이런 단순 효과가 탐지 성능을 얼마나 흔들 수 있는지 실험합니다. 이 점에서 Sunday 논문은, SONICS 수준의 복잡한 아키텍처로 넘어가기 전에 "짧은 10초 클립, Mel Spectrogram, 범용 CNN" 이라는 직관적인 조합이 어디까지 작동하는지 보여 주는 출발점 역할을 합니다.

이 리뷰에서는 Sunday가 제안한 파이프라인을 실제로 실행해 본 경험을 전제로, 이 구성이 AI Music Detection을 이해하기 위한 직관적인 시작점으로서 어느 정도까지 유효한지, 또 어떤 한계를 드러내는지를 정리합니다. 사용자 재현 실험의 구체적인 수치와 비교표는 향후 별도의 포스트에서 다룰 예정이며, 이 글에서는 원 논문이 보고하는 내용과 구조 분석에 집중합니다. 

---

> 핵심 요약
> Sunday의  「Detecting Musical Deepfakes」는 FakeMusicCaps 데이터셋과 Mel Spectrogram + ResNet18 이진 분류기를 사용해 사람vs딥페이크 음악 탐지 문제를 실험적으로 분석한 연구입니다. 논문에 따르면, 10초 단위 오디오 클립 10,746개(사람 5,373 / 딥페이크 5,373)를 학습과 평가에 사용했을 때 모든 실험에서 F1, Accuracy, Recall, Precision이 80%를 상회하는 비교적 높은 성능을 달성합니다. 또한 Deezer 연구를 참조해 피치 쉬프트와 템포 스트레치 같은 단순 조작이 탐지 성능을 얼마나 떨어뜨리는지 측정하고 여러 조작 데이터셋을 연속적으로 학습하는 Continuous Learning 설정이 사람 음악 재현율을 높이는 대신 오탐률을 크게 증가시키는 트레이드오프를 보여 줍니다.

---

## 논문 정보

|항목|내용|
|---|---|
|제목|Detecting Musical Deepfakes|
|저자|Nicholas Sunday, Department of Computer Science, The University of Texas at Austin, USA|
|소속|Department of Computer Science, The University of Texas at Austin, USA|
|유형|arXiv preprint (cs.SD, cs.LG), UT Austin coursework 기반 연구|
|논문 링크|[arXiv:2505.09633](https://arxiv.org/abs/2505.09633)|
|코드|[GitHub Repository](https://github.com/nicksunday/deepfake-music-detector)|
|주요 데이터셋|FakeMusicCaps[1], MusicLM 관련 MusicCaps[2]|
|주요 관련 연구|SONICS[3], "Detecting music deepfakes is easy but actually hard"[4]|

---

# 1. 연구 배경

## 1.1 Text-to-Music와 음악 딥페이크의 등장

논문에 따르면 최근 Text to Music 플랫폼의 발전으로 짧은 텍스트 프로프트만으로도 사람 연주와 구분하기 어려운 수준의 음악을 생성할 수 있는 환경이 빠르게 확산되고 있습니다. Sunday는 이 흐름을 이미지 딥페이크와 음성 딥페이크가 만들어낸 상황과 유사한 맥락에서 이해합니다. 플랫폼은 창작 접근성을 높이고 새로운 실험을 가능하게 하지만, 동시에 다음과 같은 문제를 동반합니다.

- 저작권 침해
- 거짓 저자 표기와 크레딧 왜곡
- 예술적 진정성에 대한 신뢰 저하

저자는 이러한 문제를 완전히 새로운 영역이라기보다, 기존 딥페이크 논의가 음악 영역으로 확장된 사례로 봅니다. 특히 "누가 이 음악을 만들었는지"와 "얼마나 사람 창작에 의존했는지"를 둘러싼 법적, 윤리적 논의가 본격화되면서, 플랫폼 차원에서 사람 음악과 생성 음악을 구분해 주는 기술이 정책과 운영 모두에서 중요해지고 있다는 점을 강조합니다.

## 1.2 FakeMusicCaps와 MusicCaps

이 실험은 Politecnico di Milano 연구진이 제안한 FakeMusicCaps 데이터셋 [1]에 기반합니다. FakeMusicCaps는 Google의 MusicCaps [2] 에서 제공하는 텍스트 설명을 바탕으로 여러 Text to Music 모델이 생성한 딥페이크 음악을 모아 구축된 데이터셋입니다.

저자가 정리한 FakeMusicCaps의 구조는 다음과 같습니다.

- 사람 음악
    - MusicCaps 기반 사람 연주 10초 오디오 클립 5,373개
- 딥페이크 음악
    - 다섯개 TTM 플랫폼(MusicGen, audioldm2, musicdm, mustango, stable_audio_open)이 생성한 10초 딥페이크 트랙 5,521개
    - Suno 플랫폼에서 생성된 더 긴 딥페이크 트랙 63개

이 가운데 저자는 딥페이크 5,373개를 무작위로 선택해 사람 음악 5,373개와 균형을 맞춘 뒤, 총 10,746개 샘플을 "사람 vs 딥페이크"이진 분류 데이터셋으로 재구성합니다. 이때 개별 TTM 플랫폼을 구분하지 않고, 모든 플랫폼을 하나의 "딥페이크" 클래스로 합쳐 버리는 설계 선택을 합니다. 이 선택 덕분에 모델 구조와 실험 설계가 단순해지는 대신, 플랫폼별 특성과 차이를 분석하는 가능성은 일부 포기하는 형태가 됩니다.

## 1.3 SONICS, Deezer 연구와 Sunday 논문의 위치

Sunday는 관련 연구로 SONICS [3]와 Deezer 연구(Afchar et al., 2024) [4]를 핵심 축으로 인용합니다.

- SONICS
    - SONICS는 Syntetic Or Not, Identifying Counterfeit Songs라는 이름 그대로 전체 곡 단위 딥페이크를 대상으로 SpecTTTra라는 새로운 아키텍처와 풀 길이 데이터셋을 제안합니다.
    - 이 연구는 짧은 클립 기반 데이터셋이 곡 구조, 가사 편곡 같은 문맥 정보를 충분히 담지 못한다고 지적하고, 풀 곡 수준에서의 구조적 다양성과 맥락 모델링을 강조합니다.

- Deezer 연구(Afchar et al., 2024)
    - Deezer연구는 Sunday가 인용한 바에 따르면, 사람 음악과 딥페이크 음악을 CNN으로 분류하는 작업은 기본 설정에서는 비교적 어렵지 않지만, 피치 쉬프트와 템포 스트레치 같은 단순 조작만으로도 탁월한 모델이 쉽게 무력화될 수 있다는 점을 보여줍니다.
    - 이 연구는 "탐지 자체가 중요한 것이 아니라 조작에 대한 견고함이 핵심 문제"라는 관점을 제시합니다.

Sunday의 Detecting Musical Deepfakes는 이 두 축 사이 어딘가에 위치합니다.

- SONICS처럼 전체 곡 단위 복잡한 아키텍처로 가지 않고
- Deezer 연구에서 중요하게 다룬 피치와 템포 조작 시나리오를 그대로 가져와
- FakeMusicCaps와 ResNet18이라는 비교적 단순한 조합이 조작에 얼마나 취약한지 다시 점검합니다.

이 관점에서 보면 Sunday 논문은, "단순한 Mel Spectrogram과 범용 CNN만으로 어느 정도까지 버틸 수 있는가"를 살펴보는 중간 단계의 정리 작업으로 이해할 수 있습니다.

## 1.4 법, 윤리, 정책 논의의 개요

논문 후반부에서 Sunday는 Text to Music 플랫폼이 만들어낼 수 있는 법적, 윤리적 이슈를 개관합니다. 논문에 따르면 대표적으로 다음과 같은 경우들이 문제로 제기됩니다.

- 기존 곡을 모사하거나, 특정 아티스트의 스타일을 과도하게 모방하는 생성물로 인한 저작권 침해 가능성
- AI가 만든 곡을 사람의 창작물인 것처럼 크레딧을 붙이고 판매하거나 공개하는 경우 발생할 수 있는 사기와 신뢰 훼손 문제
- 아티스트의 음성을 무단으로 학습해 목소리 자체를 복제하는 경우 퍼블리시티 권리 침해 가능성

동시에 Sunday는 모든 딥페이크가 악용으로 이어지는 것은 아니라는 점도 사례를 통해 보여 줍니다. 질병이나 사고로 목소리를 잃은 사람이 AI 기반 음성 복원을 통해 다시 노래를 부를 수 있게 되는 사례, 사망한 아티스트의 미완성 작업을 유족과 레이블이 함께 AI를 통해 마무리하는 사례 등은 기술의 긍정적 가능성을 보여 주는 예로 소개됩니다.

이 논문은 명시적인 법률 해석을 제시하기보다는, 이러한 사례들이 "누가 이 음악을 만들었는지"와 "어떤 맥락에서 허용 가능한지"에 대한 사회적 합의와 규범을 요구한다는 점을 강조합니다. 그리고 이런 논의의 한 축으로서 "탐지 기술"이 필요하다는 문제의식을 전면에 두고 있습니다.

## 1.5 이 리뷰의 관점

이 블로그의 관점에서 Sunday 논문은 다음과 같은 이유로 첫 리뷰 대상으로 선택되었습니다.

1. FakeMusicCaps, Mel Spectrogram, ResNet18이라는 조합이 비교적 직관적이고, 코드가 공개되어 있어 재현과 변형이 용이합니다.
2. Afchar 등 Deezer 연구와 SONICS, FakeMusicCaps 논문을 자연스럽게 엮으면서, Text to Music 딥페이크 탐지 연구의 흐름 속에서 자신을 위치시키고 있습니다.
3.짧은 10초 클립이라는 제약과 피치, 템포 조작에 대한 취약성을 동시에 보여 주기 때문에, 이후 더 복잡한 모델과 방어 기법을 다룰 때 "어디까지가 단순 모델로 가능한 영역인지"를 감각적으로 이해하는 데 도움이 됩니다.

이 리뷰는 논문이 제시하는 기술 구성과 실험 결과, 법과 윤리 논의를 가능한 한 충실히 정리하고, 향후 AI 음악 탐지 및 방어 논문 리뷰로 확장해 가기 위한 기반을 마련하는 것을 목표로 합니다.

---

# 2. 방법론 분석

## 2.1 문제 정의

저자가 정의하는 문제는 다음과 같은 이진 분류 과제입니다.

- 입력
    - 길이 10초의 음악 오디오 클립
    - FakeMusicCaps 기반 Mel Spectrogram 이미지
- 출력
    - 레이블 0: 사람 연주 음악
    - 레이블 1: TTM 기반 딥페이크 음악

 
이를 수식으로 쓰면, Mel Spectrogram을 $\mathbf{x} \in \mathbb{R}^{C \times H \times W}$, 레이블을 $y \in \{0, 1\}$라고 할 때, ResNet18 기반 분류기 $f_{\theta}$를 학습하는 문제입니다.

학습 목표는 일반적인 교차 엔트로피 손실을 최소화하는 것으로 쓸 수 있습니다.

$$
\min_{\theta} \; \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}}
\big[ \mathcal{L}_{\mathrm{CE}}(f{\theta}(\mathbf{x}), y) \big]
$$

여기서 $\mathcal{D}$는 FakeMusicCaps에서 구성한 이진 분류 데이터셋, $\mathcal{L}_{\mathrm{CE}}$는 교차 엔트로피 손실 함수입니다. 직관적으로는 Mel Spectrogram 이미지를 입력받아 살마 음악인지 딥페이크 음악인지 구분하는 이미지 분류 모델을 학습한다는 뜻입니다.

## 2.2 데이터셋 구성과 전처리

논문에 따르면 FakeMusicCaps 재구성 과정은 다음과 같습니다.

- 사람 클래스
    - MusicCaps 기반 사람 연주 10초 클립 5,373개
- 딥페이크 클래스
    - 다섯 개 TTM 플랫폼의 딥페이크 트랙에서 5,373개만 균형 샘플링
- 최종 데이터셋
    - 총 10,746개 샘플
    - 학습 8,599, 검증 1,075, 테스트 1,074 텐서로 분할

모든 오디오는 librosa를 사용해 Mel Spectrogram으로 변환되며, 이후 PyTorch 텐서로 변환하고 ImageNet1k v1 가중치에서 사용된 평균과 분산으로 정규화합니다. 이 정규화는

- 평균 [0.485, 0.456, 0.406]
- 표준편차 [0.299, 0.224, 0.225]

를 사용합니다. ResNet18을 그대로 활용하기 위해 기본 ImageNet 전처리 파이프라인을 그대로 가져온 셈입니다.

논문에 따르면 실험에 사용된 네 가지 데이터셋은

1. Base
    - 조작하지 않은 원본 FakeMusicCaps
2. Pitch
    - 무작위 반음 단위 피치 쉬프트 적용
3. Tempo
    - 무작위 비율(0.8에서 1.2 사이)로 템포 스트레치 또는 압축 적용
4. PitchTempo
    - 피치 쉬프트와 템포 스트레치를 모두 적용

으로 정의됩니다. Pitch와 Tempo 변조에는 모두 librosa의 시그널 처리 함수가 사용되며, 각 클립마다 서로 다른 난수가 샘플링됩니다.

## 2.3 모델 구조와 학습 설정

모델 측면에서 Sunday는 ResNet18을 선택합니다. 논문에 따르면

- ResNet18
    - torchvision에서 제공하는 ImageNet1k v1 사전학습 가중치 사용
    - 마지막 완전연결층을 출력 차원 2로 교체
    - 손실 함수는 Cross Entropy
    - 옵티마이저는 Adam

학습 설정은 다음과 같습니다.

    - 배치 크기 32
    - 에포크 20
    - 학습과 검증, 테스트 데이터 분할 비율은 위에서 언급한 텐서 수 기반

별도의 오디오 전용 아키텍처나 특수한 정규화 기법을 사용하지 않고, Mel Spectrogram과 범용 CNN이라는 가장 직관적인 구성을 그대로 가져왔습니다. 이 선택은 이후 더 복잡한 아키텍처를 도입할 떄 비교 기준으로 사용하기에 적합한 동시에, 과도한 구조적 복잡성 없이 데이터 자체의 특성을 탐색하기 쉽다는 장점이 있습니다. 

## 2.4 악의적 조작과 실험 설계

논문에 따르면 이 연구는 Deezer 연구에서 제시된 아이디어를 따라, 조작된 딥페이크가 탐지 모델을 얼마나 쉽게 속일 수 있는가를 실험적으로 평가하는 데 초점을 맞춥니다. 구체적으로는

- 피치 쉬프트
    - 각 클립마다 -2에서 2사이의 난수를 샘플링
    - 해당 값만큼 반음 단위로 피치를 이동
- 템포 스트레치 
    - 각 클립마다 0.8에서 1.2 사이의 난수를 샘플링
    - 이 비율만큼 재생 속도를 늘리거나 줄여 새로운 클립 생성

이렇게 생성된 Pitch, Tempo, PitchTempo 데이터셋에 대해

- 단일 데이터셋별 학습 네 가지
- 네 데이터셋을 순차적으로 학습하는 Continuous Learning 한 가지

총 다섯 가지 설정으로 ResNet18 모델을 학습합니다. Continuous Learning 실험은 실제 악의적 공격 시나리오에서 모델이 다양한 조작 유형을 순차적으로 경험했을 때, 사람 음악과 딥페이크 음악을 각각 얼마나 정확하게 식별하는지, 그리고 그 과정에서 어떤 성능 트레이드 오프가 발생하는지 확인하기 위한 시도라고 볼 수 있습니다.

---

## 3. 실험 재현 

이 절에서는 저자가 공개한 GitHub 레포지토리를 기반으로 실험을 재현하면서 확인한 사항과, 블로그 독자가 같은 실험을 따라 해 볼 때 위의할 포인트를 정리합니다. 구체적인 수치 결과는 작성자 환경에서의 실험이 더 축적된 뒤 별도 글에서 자세히 다룰 예정이며, 여기서는 재현 과정과 설정 위주로 정리합니다.

## 3.1 실험 환경 메모

아래 표는 논문과 GitHub 레포지토리 기준의 설정과, 작성자가 재현할 때 참고한 구성을 비교해 정리한 것입니다. 작성자 측 세부 환경은 실제 실험을 마친 뒤 채워 넣는 것을 전제로 비워 두었습니다.

| 항목 | 원 논문 기준 설정 |
|-----|---------------|
| 프로그래밍 언어 | Python, PyTorch, librosa 사용 언급 |
| 입력 표현 | Mel Spectrogram, ImageNet 정규화 |
| 모델 | ResNet18, ImageNet1k v1 가중치, 출력 차원2 |
| 데이터 분할 | 학습 8,599 검증 1,075 테스트 1,074 |
| 하이퍼파라미터 | 배치 32, 에포크 20, Adam, Cross Entropy |
| 하드웨터 | 논문에 구체적 하드웨어 미기재 |

실제 재현 시에는 GitHub 레포지토리의 notebooks 디렉터리에 있는 데이터셋 생성 및 학습 노트북을 그대로 실행하는 방식으로 시작하는 것이 가장 자연스럽습니다. 이후 실험을 확장하고 싶다면, 노트북에서 주요 파이프라인만 함수화해 별도 모듈로 옮기는 식으로 리팩토링하는 것이 좋습니다.

## 3.2 원 논문 결과와 재현 계획

논문은 각 데이터셋과 실험 설정에 대해 Accuracy, Precision, Recall, F1, False Positive Rate, False Negative Rate를 보고합니다. 논문에 따르면

- Base 데이터셋에서 학습한 모델이 전반적인 지표에서 가장 안정적인 성능을 보이며
- Pitch, Tempo, PitchTempo 데이터셋에 대해서는 각기 몇 퍼센트포인트 수준의 성능 저하가 관잘되고
- 네 데이터셋을 순차적으로 학습한 Continuous Learning 모델은 사람 음악 Recall과 False Negative Rate에서 가장 좋은 수치를 기록하지만, False Positive Rate가 다른 모델보다 크게 상승하는 트레이드 오프를 보입니다.

이 블로그 글에서는 원 논문 수치를 그대로 복제해 나열하기보다는, 재현 실험이 어느 정도 축적된 뒤 

- 원 논문 수치와 작성자 재현 수치를 나란히 비교하는 표
- 피치와 템포 조작 강도에 따른 성능 변화 곡선
- Coninuous Learning과 단일 데이터셋 학습 간 혼동 행렬 비교

를 별도 포스트에서 다루는 것을 목표로 합니다. 따라서 여기서는 조작이 강해질수록 탐지가 더 어려워지고, 조작 유형에 따라 난이도가 다르다는 논문의 정성적 결론만 정리해 두었습니다.

## 3.3 구현 핵심 코드 스니펫


---
아래 번호는 Sunday 논문 참고문헌과 정합성을 최대한 맞추어 정리했습니다.

[1] Comanducci, L., Bestagini, P., and Tubaro, S. (2024). FakeMusicCaps: a Dataset for Detection and Attribution of Synthetic Music Generated via Text-to-Music Models. arXiv:2409.10684.

[2] Agostinelli, A., Denk, T. I., Borsos, Z., Engel, J., Verzetti, M., Caillon, A., and others. (2023). MusicLM: Generating Music from Text. arXiv:2301.11325.

[3] Rahman, M. A., Hakim, Z. I. A., Sarker, N. H., Paul, B., and Fattah, S. A. (2024). SONICS: Synthetic Or Not – Identifying Counterfeit Songs. arXiv:2408.14080.

[4] Afchar, D., Meseguer-Brocal, G., and Hennequin, R. (2024). Detecting music deepfakes is easy but actually hard. arXiv:2405.04181.