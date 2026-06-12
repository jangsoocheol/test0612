# 🧬 하디-바인베르크 & 진화 탐구 실험실 — 설계 문서

## 1. 전체 시스템 아키텍처

```
hw_evolution_lab/
├── app.py              ← Streamlit UI 메인 (탭 4개)
├── simulation.py       ← 시뮬레이션 엔진 (H-W / 진화 모델 분리)
├── charts.py           ← Plotly 시각화 모듈
├── requirements.txt
└── ARCHITECTURE.md     ← 이 파일
```

```
┌─────────────────────────────────────────────────────────────┐
│                         app.py (UI)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │TAB 1:    │ │TAB 2:    │ │TAB 3:    │ │TAB 4:        │   │
│  │탐구 설계  │ │결과 그래프│ │이론 vs   │ │결론 & 보고서  │   │
│  │(가설입력) │ │          │ │실제 비교  │ │(텍스트 작성)  │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │
│         ↑                        ↑                           │
│    session_state (df_hw, df_evo, hypothesis, …)             │
└─────────────────────────────────────────────────────────────┘
          ↓ run_experiment()
┌─────────────────────────────────────────────────────────────┐
│                      simulation.py                          │
│  ┌──────────────────────┐   ┌──────────────────────────┐   │
│  │  HardyWeinbergModel  │   │     EvolutionModel        │   │
│  │  (이론 기준선)        │   │  (자연선택+돌연변이+부동)  │   │
│  │  p = 일정, 오차 = 0  │   │  매 세대 3단계 갱신       │   │
│  └──────────────────────┘   └──────────────────────────┘   │
│            ↓ GenerationRecord ↓                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │       compute_deviation_summary() → 오차 DataFrame   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          ↓ DataFrames
┌─────────────────────────────────────────────────────────────┐
│                        charts.py                            │
│  plot_allele_frequency()     plot_genotype_frequency()      │
│  plot_hw_vs_evo_comparison() plot_error()                   │
│  plot_mean_fitness()                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 데이터 구조 설계

### GenerationRecord (dataclass)
```python
generation: int       # 세대 번호 (0~N)
p: float              # A 대립유전자 빈도
q: float              # a 대립유전자 빈도  ← q = 1-p
freq_AA: float        # AA 유전자형 실제 빈도
freq_Aa: float        # Aa 유전자형 실제 빈도
freq_aa: float        # aa 유전자형 실제 빈도
hw_AA: float          # AA H-W 이론 빈도 = p²
hw_Aa: float          # Aa H-W 이론 빈도 = 2pq
hw_aa: float          # aa H-W 이론 빈도 = q²
error_AA/Aa/aa: float # |이론 - 실제|
mean_fitness: float   # w̄ = p²wAA + 2pqwAa + q²waa
```

### SimulationConfig (dataclass)
```python
p0: float             # 초기 A 빈도 (조작 변수)
w_AA, w_Aa, w_aa: float  # 유전자형별 적응도
mu_Aa: float          # A→a 돌연변이율 μ
mu_aA: float          # a→A 돌연변이율 ν
population_size: int  # None이면 무한대 (부동 없음)
generations: int      # 시뮬레이션 세대 수
```

---

## 3. 시뮬레이션 알고리즘

### H-W 이론 모델
```
초기: p = p0, q = 1-p0
매 세대: freq_AA = p², freq_Aa = 2pq, freq_aa = q²
         p는 변하지 않음 (평형 조건)
```

### 진화 모델 (매 세대 3단계)
```
Step 1 - 자연선택:
  p_new = (p²·wAA + pq·wAa) / w̄
  w̄ = p²·wAA + 2pq·wAa + q²·waa

Step 2 - 돌연변이:
  p_new = p·(1-μ) + (1-p)·ν

Step 3 - 유전적 부동 (population_size ≠ None):
  alleles_A ~ Binomial(2N, p)
  p_new = alleles_A / (2N)
```

---

## 4. 탐구 활동 시나리오 (수업 설계용)

### 시나리오 A — H-W 평형 조건 확인 (탐구 1)
- 설정: w=1.0/1.0/1.0, μ=0, N=∞, p₀=0.3
- 예상: p 변화 없음, 오차 = 0
- 탐구 질문: "왜 p가 변하지 않는가? 어떤 조건이 이것을 가능하게 하는가?"

### 시나리오 B — 방향 선택 (탐구 2)
- 설정: w(AA)=1.5, w(Aa)=1.0, w(aa)=0.5, N=∞, p₀=0.3
- 예상: p → 1 수렴, aa 소멸
- 수학 연결: p' = f(p) 그래프, 고정점 분석

### 시나리오 C — 초열성 (잡형 유리) (탐구 3)
- 설정: w(AA)=0.8, w(Aa)=1.2, w(aa)=0.8, p₀=0.2
- 예상: p가 중간값으로 수렴 (다형성 유지)
- 생명과학 연결: 낫모양 적혈구 빈혈증 설명

### 시나리오 D — 유전적 부동 (탐구 4)
- N=30 vs N=500, 여러 랜덤 시드 반복
- 예상: 소집단에서 고정 발생, 대집단은 H-W 유지
- 수학 연결: 이항분포, 분산 = 2Npq

### 시나리오 E — 돌연변이 평형 (탐구 5)
- μ=0.001, ν=0.0001, w=1/1/1
- 예상: p → ν/(μ+ν) = 0.0909 수렴
- 수학 연결: 수열 점화식, 극한값 계산

---

## 5. 향후 확장 가능 구조

### Phase 2 확장 목록
- [ ] 복수 유전자 위치 (2-locus) 지원
- [ ] 유전자 흐름 (Gene Flow) 모델 추가
- [ ] 집단 간 비교 (다집단 시뮬레이션)
- [ ] 환경 변화 이벤트 (세대마다 w 변경)
- [ ] 실험 결과 저장 & 비교 기능 (여러 실험 overlay)
- [ ] 교사 대시보드 (학생 가설 수집 & 전체 결과 비교)

### 코드 확장 포인트
```python
# EvolutionModel.run()에 환경 변화 훅 추가 예시
class EvolutionModel:
    def run(self, environment_events: dict[int, dict] = None):
        for gen in range(generations):
            if environment_events and gen in environment_events:
                # 예: {20: {"w_AA": 0.5, "w_aa": 1.5}} → 20세대에 환경 반전
                self.cfg = update_config(self.cfg, environment_events[gen])
```

---

## 6. MVP 체크리스트

- [x] HardyWeinbergModel (기준 이론 모델)
- [x] EvolutionModel (자연선택 + 돌연변이 + 부동)
- [x] 가설 직접 입력 UI
- [x] 두 모델 동시 실행 및 비교
- [x] p/q 변화 그래프
- [x] 유전자형 빈도 그래프 (이론/실제 분리)
- [x] 이론 vs 실제 막대 비교 그래프
- [x] 오차 그래프
- [x] 평균 적응도 그래프
- [x] 탐구 보고서 다운로드
- [x] 교사용 수업 가이드 내장
