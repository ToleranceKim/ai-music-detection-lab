# Detecting Musical Deepfakes

- title: "[논문 리뷰] Detecting Musical Deepfakes – FakeMusicCaps와 ResNet18 기반 AI 음악 딥페이크 탐지"
- date: 2025-12-05
- tags: [AI Music Detection, Deepfake, FakeMusicCaps, ResNet18]
---

# 리뷰를 시작하며

현재 AI 생성 음악을 둘러싼 연구는 생성 음악 탐지, 워터마킹과 인증, 데이터 포이즈닝 공격과 방어 등 여러 방향에서 진행되고 있습니다. 

이 블로그에서는 그 가운데 특히 (1) 플랫폼이나 서비스 단계에서 생성 음악을 식별하는 AI Music Detection과 (2) 학습 데이터 자체를 방어하려는 Unlearnable Data와 Defensive Data Poisoning 두 축에 초점을 맞추어 논문을 리뷰할 예정입니다. AI 생성 음악이 음악 생태계에 던지는 더 넓은 질문에 대한 저의 생각과 이 블로그의 문제의식은 별도의 글에서 다룰 예정입니다.

오늘은 최신 AI Music Detection 연구의 성과를 직관적으로 엿볼 수 있는 논문을 통해 생성 음악 식별 기술을 체험하고 분석하는데 집중해보겠습니다.

오늘 다룰 Nicholas Sunday의 「Detecting Musical Deepfakes」는 FakeMusicCaps 기반 오디오를 멜 스펙트로그램으로 변환하고 ResNet18 분류기를 학습하는 직관적인 베이스라인을 제안합니다. 전체 파이프라인이 GitHub에 공개되어 있어 실험 재현과 변형 시도가 용이하고, Afchar 등 Deezer 연구와 SONICS, FakeMusicCaps 논문을 주요 선행 연구로 검토한 뒤 Afchar 논문에서 다룬 피치 쉬프트와 템포 스트레치 조작 시나리오를 FakeMusicCaps와 ResNet18 구성에서 다시 실험한다는 점에서, 이후 AI Music Detection 시리즈에서 다룰 보다 복잡한 모델 방어 기법을 이해하기 위한 기준선으로 적합하다고 판단했습니다.

이 리뷰에서는 공개 코드를 직접 재현하고 실행한 경험을 바탕으로, Sunday가 제안한 설정이 어느 지점에서 유효하고 어떤 부분에서 한계를 보이는지 확인합니다.


# Detecting Musical Deepfakes: FakeMusicCaps와 ResNet18으로 살펴본 음악 딥페이크 탐지
> Sunday의  「Detecting Musical Deepfakes」는 FakeMusicCaps 데이터셋과 Mel Spectrogram + ResNet18 이진 분류기를 사용해 사람vs딥페이크 음악 탐지 문제를 실험적으로 분석한 연구입니다. 논문에 따르면, 10초 단위 오디오 클립 10,746개(사람 5,373 / 딥페이크 5,373)를 학습과 평가에 사용했을 때 모든 실험에서 F1, Accuracy, Recall, Precision이 80%를 상회하는 비교적 높은 성능을 달성합니다. 또한 Deezer 연구를 참조해 피치 쉬프트와 템포 스트레치 같은 단순 조작이 탐지 성능을 얼마나 떨어뜨리는지 측정하고 여러 조작 데이터셋을 연속적으로 학습하는 Continuous Learning 설정이 사람 음악 재현율을 높이는 대신 오탐률을 크게 증가시키는 트레이드오프를 보여 줍니다.

---

## 논문 정보 (Paper Information)

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

최근 Text-to-Music(TTM) 플랫폼이 빠르게 발전하면서 짧은 텍스트 프롬프트만으로도 고품질 음악을 생성할 수 있게 되었습니다. 논문에서는 MusicLM 계열 연구[2]와 상용 또는 오픈소스 TTM 모델의 보급이 가져오는 변화를 다음과 같이 제시합니다.

- 전문 음악 지식 없이도 빠르게 배경음악, 짧은 트랙, 사운드스케이프를 생성할 수 있음
- 특정 아티스트의 스타일을 모사하거나 기존 곡을 변형한 딥페이크가 대량 생산될 위험이 커짐
- 로열티를 노린 합성 음원이 스트리밍 서비스에 대량 업로드될 가능성이 존재함
- 유명 아티스트의 목소리와 이미지를 모사한 딥페이크, 사후 인물을 가상으로 재현하는 사례가 늘어나면서 법적, 윤리적 논쟁이 심화됨[16]

Sunday는 Drake의 인공지능 곡 논란이나 Tupac Shakur의 홀로그램 공연 사례 등을 인용하면서 음악 딥페이크가 단순한 기술 데모 수준을 넘어서 실제 산업과 법제, 팬 커뮤니티의 신뢰에 영향을 미치는 문제로 확장되고 있다고 설명합니다.
---

# 1.2 기존 연구와 데이터셋의 한계

논문은 선행 연구를 크게 세 축으로 정리합니다.

1. **FakeMusicCaps 데이터셋**[1]<br>
MusicCaps[2]의 인간 작성 텍스트 캡션을 기반으로 여러 TTM 모델(MusicGen_medium, AudioLDM2, MusicLDM, Mustango, Stable Audio Open 등)을 사용해 딥페이크 음악을 생성한 데이터셋입니다. Comanducci 등[1]에 따르면 FakeMusicCaps는 딥페이크 탐지뿐 아니라 생성 플랫폼 귀속(attribution)까지 실험할 수 있도록 설계되었습니다.

2. **SONICS(Synthetic or Not - Identifying Counterfeit Songs)**[3]<br>
Rahman 등[3]에 따르면 기존 딥페이크 음악 데이터셋은 보통 짧은 클립 위주이고 장르와 가사 다양성이 부족하며 곡 전체 단위 위조가 충분하지 않은 한계를 갖습니다. SONICS는 곡 전체 길이의 위조 곡과 가사를 포함한 데이터셋, SpecTTTra 아키텍처를 제안하면서 보다 현실적인 위조 탐지 프레임워크를 구축합니다.

3. **Deezer의 "Detecting music deepfakes is easy but actually hard"**[4]<br>
Afchar 등[4]에 따르면 기본적인 살마 대 딥페이크 구분은 CNN 기반 모델로 비교적 쉽게 높은 성능을 얻을 수 있지만 피치 쉬프트와 템포 변화 같은 간단한 후처리만으로도 탐지 성능이 크게 떨어질 수 있습니다. 다시 말해

Sunday의 논문은 특히 Deezer[4]와 FakeMusicCaps[1]를 바탕으로 Mel Spectrogram과 ResNet18 구조 에서 유사한 현상이 재현되는지 탐색하는 것을 목표로 합니다.

# 1.3 연구 질문과 기여

---
아래 번호는 Sunday 논문 참고문헌과 정합성을 최대한 맞추어 정리했습니다.

[1] Comanducci, L., Bestagini, P., and Tubaro, S. (2024). FakeMusicCaps: a Dataset for Detection and Attribution of Synthetic Music Generated via Text-to-Music Models. arXiv:2409.10684.

[2] Agostinelli, A., Denk, T. I., Borsos, Z., Engel, J., Verzetti, M., Caillon, A., and others. (2023). MusicLM: Generating Music from Text. arXiv:2301.11325.

[3] Rahman, M. A., Hakim, Z. I. A., Sarker, N. H., Paul, B., and Fattah, S. A. (2024). SONICS: Synthetic Or Not – Identifying Counterfeit Songs. arXiv:2408.14080.

[4] Afchar, D., Meseguer-Brocal, G., and Hennequin, R. (2024). Detecting music deepfakes is easy but actually hard. arXiv:2405.04181.