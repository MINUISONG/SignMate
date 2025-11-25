# 🤟 SignMate: AI-based Real-time Sign Language Tutor

> **Gachon University P-Practical Project (Graduation Project) Team 8**
> **"Bridging the Gap in the Critical Period: An AI Tutor Connecting Parents and Children"**

---

## 🌍 English Description

### 1. 🌟 Project Background

**"90% of deaf children are born to hearing parents."**
For deaf children, the period before age 5 is the **'Critical Period'** for language development. Without proper language stimulation during this time, they risk falling into a state of 'Language Deprivation,' leading to irreversible damage to cognitive functions.

However, most hearing parents do not know sign language, and the existing education market is limited to **'boring one-way videos'** or **'simple dictionaries,'** causing many to miss this golden time for learning.

**SignMate** solves this problem with an AI solution that goes beyond simple learning. It **Assesses** whether your movements are correct in real-time and provides **Coaching** like a teacher to correct mistakes.

### 2. 💡 Key Features

| Feature | Description |
| :--- | :--- |
| **Interactive Learning** | Provides an active learning environment with **Gamification** (quizzes, games) to eliminate boredom. |
| **Ghost Overlay UI** | Overlays a semi-transparent 'Answer Skeleton (Ghost)' on the user's screen to induce intuitive posture correction. |
| **Hybrid Feedback** | A 3-stage feedback system: **Rule-based (Instant)** + **Deep Learning (Precise)** + **LLM (Natural Language)**. |

### 3. 🛠️ Technical Pipeline

This project establishes a 3-stage hybrid pipeline to ensure both real-time performance and accuracy.

<img width="779" height="318" alt="image" src="https://github.com/user-attachments/assets/d816f8bb-5911-4284-84be-bec1193c3dba" />

#### Phase 1. Real-time Sensing & Geometric Feedback (Instant Correction)
* **Vision AI:** Extracts 543 3D keypoints (Hands: 42, Pose: 33, Face: 468) in real-time using `MediaPipe Holistic`.
* **Geometric Heuristics:** Calculates angles and positions of major joints using **Vector Arithmetic**.
* **DTW (Dynamic Time Warping):** Accurately calculates time-series similarity and performs initial scoring even if the speed of the user and the reference video differs.

#### Phase 2. Deep Linguistic Analysis (Deep Analysis)
* **Linguistic Slicing:** Separates total keypoints into 4 key elements of sign language (**①Handshape, ②Location, ③Movement, ④NMS**) for independent analysis.
* **Feature Encoding:** Compresses long sequences into semantic feature vectors using `MS-TCN` (Multi-Stage TCN).
* **AQA (Action Quality Assessment):** Aligns User and Ground Truth (GT) sequences using **Cross-Attention** mechanisms and calculates precise error scores (JSON) for each component.

#### Phase 3. Generative Coaching (LLM Feedback)
* **Input:** Quantitative score data produced in Phase 2 (e.g., `{"handshape_score": 55, "error_loc": "T3"}`).
* **LLM Processing:** Analyzes data using `Gemini` or `GPT` API to generate feedback in a warm, encouraging tone (e.g., *"Your hand shape was correct, but your wrist dropped a bit in the middle. Let's try raising it again?"*).

### 4. 💾 Dataset



### 5. 🏗️ Tech Stack

* **AI Model:** Python, PyTorch, MediaPipe, MS-TCN, Transformer (Cross-Attention)
* **Algorithm:** DTW (Dynamic Time Warping), Cosine Similarity
* **Backend:**
* **LLM:**
* **Frontend:** 

---

## 🇰🇷 Korean Description

### 1. 🌟 Project Background (연구 배경)

**"90%의 청각장애 아동은 청인(비장애인) 부모에게서 태어납니다."**
청각장애 아동에게 만 5세 이전은 언어 발달의 **'결정적 시기(Critical Period)'**입니다. 이 시기에 적절한 언어 자극을 받지 못하면 '언어 박탈(Language Deprivation)' 상태에 빠져 인지 기능 전반에 돌이킬 수 없는 손상을 입게 됩니다.

하지만 대부분의 청인 부모는 수어를 모르며, 기존 교육 시장은 **'지루한 일방향 강의'**나 **'단순 사전'**에 머물러 있어 학습의 골든타임을 놓치게 만듭니다.

**SignMate**는 이러한 문제를 해결하기 위해, 단순한 학습을 넘어 **"내가 한 동작이 맞는지 즉시 알려주고, 틀린 부분을 선생님처럼 교정해주는"** AI 솔루션입니다.

### 2. 💡 Key Features (핵심 기능)

| Feature | Description |
| :--- | :--- |
| **Interactive Learning** | 퀴즈와 게임(Gamification) 요소를 도입하여 지루함을 없앤 능동적 학습 환경 제공 |
| **Ghost Overlay UI** | 사용자 화면 위에 '정답 Ghost' 반투명하게 겹쳐 직관적인 자세 교정 유도 |
| **Hybrid Feedback** | **규칙 기반(즉각적)** + **딥러닝(정밀함)** + **LLM(자연어)**의 3단계 피드백 시스템 |


### 3. 🛠️ Technical Pipeline (기술 아키텍처)

본 프로젝트는 실시간성과 정확도를 동시에 확보하기 위해 3단계 하이브리드 파이프라인을 구축했습니다.

<img width="779" height="318" alt="image" src="https://github.com/user-attachments/assets/1aa97188-e71c-462a-b747-7ce3d936e398" />

#### Phase 1. Real-time Sensing & Geometric Feedback (즉각 교정)
* **Vision AI:** `MediaPipe Holistic`을 통해 손(42), 몸(33), 얼굴(468)의 총 543개 3D 키포인트를 실시간 추출합니다.
* **Geometric Heuristics:** **벡터 연산(Vector Arithmetic)**으로 주요 관절의 각도와 위치를 계산합니다.
* **DTW (Dynamic Time Warping):** 사용자와 정답 영상의 속도가 달라도 시계열 유사도를 정확히 계산하여 1차 채점을 수행합니다.

#### Phase 2. Deep Linguistic Analysis (심층 언어학적 분석)
* **Linguistic Slicing:** 전체 키포인트를 수어의 4대 요소(**①수형, ②수위, ③수동, ④비수지**)로 분리하여 독립 분석합니다.
* **Feature Encoding:** `MS-TCN` (Multi-Stage TCN) 모델을 사용하여 긴 시퀀스를 의미론적 특징 벡터로 압축합니다.
* **AQA (Action Quality Assessment):** **Cross-Attention** 메커니즘을 통해 사용자(User)와 정답(GT) 시퀀스를 정렬하고, 구성 요소별 정밀 오차 점수(JSON)를 산출합니다.

#### Phase 3. Generative Coaching (LLM 코칭)
* **Input:** Phase 2에서 산출된 정량적 점수 데이터 (예: `{"handshape_score": 55, "error_loc": "T3"}`)
* **LLM Processing:** `Gemini` 또는 `GPT` API를 활용하여 데이터를 분석하고, **"손 모양은 정확했지만, 중간에 손목이 조금 내려갔네요. 다시 올려볼까요?"**와 같은 따뜻한 격려 말투의 피드백을 생성합니다.

### 4. 💾 Dataset (데이터셋)



### 5. 🏗️ Tech Stack (기술 스택)

* **AI Model:** Python, PyTorch, MediaPipe, MS-TCN, Transformer (Cross-Attention)
* **Algorithm:** DTW (Dynamic Time Warping), Cosine Similarity
* **Backend:** 
* **LLM:** 
* **Frontend:** 

---

## 👥 Team 8 (Contributors)

| Role | Name | GitHub |
| :--- | :--- | :--- |
| **AI Research** | Minui Song (송민의) | [@username](https://github.com/) |
| **AI Research** | Name (이름) | [@username](https://github.com/) |
| **AI Research** | Name (이름) | [@username](https://github.com/) |
| **Backend & Eng.** | Name (이름) | [@username](https://github.com/) |
| **Backend & Eng.** | Name (이름) | [@username](https://github.com/) |
