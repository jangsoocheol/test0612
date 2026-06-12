"""
app.py  ─  하디-바인베르크 & 진화 탐구 실험실
Streamlit 메인 파일
"""

import streamlit as st
import pandas as pd
import numpy as np

from simulation import SimulationConfig, run_experiment, compute_deviation_summary
from charts import (
    plot_allele_frequency,
    plot_genotype_frequency,
    plot_hw_vs_evo_comparison,
    plot_error,
    plot_mean_fitness,
)

# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────

st.set_page_config(
    page_title="HW 진화 탐구 실험실",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# 전역 CSS
# ─────────────────────────────────────────

st.markdown("""
<style>
  /* 폰트 */
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

  /* 헤더 배너 */
  .lab-header {
    background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    color: white;
  }
  .lab-header h1 { font-size: 2rem; font-weight: 700; margin: 0 0 4px 0; }
  .lab-header p  { font-size: 0.95rem; margin: 0; opacity: 0.8; }

  /* 카드 */
  .step-card {
    background: #F8FAFC;
    border: 1.5px solid #E2E8F0;
    border-left: 5px solid #6366F1;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
  }
  .step-card h3 { margin: 0 0 6px 0; font-size: 1.05rem; color: #312E81; }
  .step-card p  { margin: 0; color: #475569; font-size: 0.92rem; }

  /* 결과 배지 */
  .badge-hw  { background:#EEF2FF; color:#4338CA; border-radius:6px; padding:4px 10px;
               font-size:0.85rem; font-weight:600; display:inline-block; }
  .badge-evo { background:#FEF3C7; color:#B45309; border-radius:6px; padding:4px 10px;
               font-size:0.85rem; font-weight:600; display:inline-block; }

  /* 가설 박스 */
  .hypothesis-box {
    border: 2px dashed #A5B4FC;
    border-radius: 10px;
    padding: 16px 20px;
    background: #EEF2FF;
    margin-bottom: 12px;
  }
  .hypothesis-box h4 { color: #3730A3; margin: 0 0 6px 0; }

  /* 결론 박스 */
  .conclusion-box {
    border: 2px solid #6EE7B7;
    border-radius: 10px;
    padding: 16px 20px;
    background: #ECFDF5;
    margin-top: 12px;
  }

  /* 사이드바 섹션 제목 */
  .sidebar-section {
    background: #312E81;
    color: white;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.88rem;
    font-weight: 600;
    margin: 16px 0 10px 0;
    letter-spacing: 0.04em;
  }

  /* 경고 배너 */
  .warn-box {
    background: #FFF7ED;
    border: 1.5px solid #FDBA74;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #9A3412;
  }

  /* 탭 강조 */
  [data-testid="stTab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 헤더
# ─────────────────────────────────────────

st.markdown("""
<div class="lab-header">
  <h1>🧬 하디-바인베르크 & 진화 탐구 실험실</h1>
  <p>수학적 모델(H-W 법칙)과 실제 진화 과정의 차이를 직접 탐구하는 과학 실험실입니다.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 사이드바: 모델 빌더
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 진화 모델 빌더")
    st.caption("아래 설정을 조작하여 나만의 진화 실험을 설계하세요.")

    # ── 초기 조건
    st.markdown('<div class="sidebar-section">📌 초기 조건</div>', unsafe_allow_html=True)
    p0 = st.slider(
        "초기 A 대립유전자 빈도 (p₀)",
        min_value=0.01, max_value=0.99, value=0.5, step=0.01,
        help="0에 가까울수록 A 유전자가 희귀, 1에 가까울수록 A 유전자가 우세합니다."
    )
    st.caption(f"q₀ = {1 - p0:.2f}  |  H-W 예측 → AA={p0**2:.3f}, Aa={2*p0*(1-p0):.3f}, aa={(1-p0)**2:.3f}")

    generations = st.slider("시뮬레이션 세대 수", 10, 200, 50, 5)

    # ── 자연선택 (적응도)
    st.markdown('<div class="sidebar-section">⚔️ 자연선택 (Fitness)</div>', unsafe_allow_html=True)
    st.caption("각 유전자형의 적응도를 설정합니다. H-W 조건 = 모두 1.0")

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        w_AA = st.number_input("w(AA)", value=1.0, min_value=0.0, max_value=2.0, step=0.05, format="%.2f")
        w_aa = st.number_input("w(aa)", value=1.0, min_value=0.0, max_value=2.0, step=0.05, format="%.2f")
    with col_w2:
        w_Aa = st.number_input("w(Aa)", value=1.0, min_value=0.0, max_value=2.0, step=0.05, format="%.2f")

    if w_AA == w_Aa == w_aa == 1.0:
        st.markdown('<div class="warn-box">⚖️ 현재 자연선택 없음 → H-W 조건 충족</div>', unsafe_allow_html=True)

    # ── 돌연변이
    st.markdown('<div class="sidebar-section">🔀 돌연변이율</div>', unsafe_allow_html=True)
    mu_Aa = st.number_input(
        "μ (A→a, 정방향 돌연변이율)",
        value=0.0, min_value=0.0, max_value=0.1, step=0.001, format="%.4f",
        help="A 대립유전자가 a로 변이될 확률 (세대당)"
    )
    mu_aA = st.number_input(
        "ν (a→A, 역방향 돌연변이율)",
        value=0.0, min_value=0.0, max_value=0.1, step=0.001, format="%.4f",
        help="a 대립유전자가 A로 변이될 확률 (세대당)"
    )

    # ── 유전적 부동
    st.markdown('<div class="sidebar-section">🎲 유전적 부동</div>', unsafe_allow_html=True)
    use_drift = st.checkbox("유전적 부동 활성화 (유한 집단)")
    population_size: int | None = None
    if use_drift:
        population_size = st.slider(
            "집단 크기 (N)",
            min_value=10, max_value=10000, value=100,
            help="집단이 작을수록 부동 효과가 강해집니다."
        )
        rng_seed = st.number_input("랜덤 시드 (재현용)", value=42, step=1)
    else:
        rng_seed = 42

    # ── 실행 버튼
    st.markdown("---")
    run_btn = st.button("▶ 시뮬레이션 실행", type="primary", use_container_width=True)


# ─────────────────────────────────────────
# 세션 상태 초기화
# ─────────────────────────────────────────

if "df_hw" not in st.session_state:
    st.session_state["df_hw"] = None
    st.session_state["df_evo"] = None
    st.session_state["config"] = None


# ─────────────────────────────────────────
# 탐구 워크플로 탭
# ─────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 탐구 설계",
    "📈 시뮬레이션 결과",
    "📊 이론 vs 실제 비교",
    "📝 결론 & 보고서",
])


# ══════════════════════════════════════════
# TAB 1: 탐구 설계 (가설 입력)
# ══════════════════════════════════════════

with tab1:
    st.markdown("## 1️⃣ 탐구 설계")
    st.info("시뮬레이션을 실행하기 **전에** 아래를 먼저 작성하세요. 과학적 탐구는 **가설 설정 → 실험 → 검증** 순서로 진행됩니다.")

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("### 🔬 탐구 문제 설정")

        research_question = st.text_area(
            "탐구 문제 (Research Question)",
            placeholder="예: 자연선택이 작용할 때 열성 유전자의 빈도는 어떻게 변할까?\n예: 집단 크기가 작아지면 H-W 평형이 얼마나 빨리 깨질까?",
            height=100,
            key="rq",
        )

        st.markdown("### 💡 나의 가설")
        st.caption("**가설은 시스템이 평가하지 않습니다.** 자유롭게 예측해 보세요.")

        hypothesis = st.text_area(
            "가설 입력 (Hypothesis)",
            placeholder=(
                "예: AA의 적응도가 aa보다 높을수록, 세대가 진행될수록 aa의 빈도는 0에 가까워질 것이다.\n"
                "예: 집단 크기가 50 이하일 때 대립유전자 고정이 100세대 이내에 발생할 것이다."
            ),
            height=130,
            key="hypo",
        )

        independent_var = st.text_input(
            "조작 변수 (Independent Variable)",
            placeholder="예: AA의 적응도 w(AA) 값",
            key="ivar",
        )
        dependent_var = st.text_input(
            "종속 변수 (Dependent Variable)",
            placeholder="예: a 대립유전자 빈도(q)의 세대별 변화",
            key="dvar",
        )
        controlled_var = st.text_area(
            "통제 변수 (Controlled Variables)",
            placeholder="예: 초기 p₀=0.5, 돌연변이 없음, 집단 크기 무한대",
            height=70,
            key="cvar",
        )

    with col_b:
        st.markdown("### 📐 H-W 법칙 이론값 미리 계산")
        q0 = 1 - p0
        st.markdown(f"""
<div class="step-card">
  <h3>현재 초기값 (p₀ = {p0:.2f})</h3>
  <p>
    ◼ AA 이론 빈도: <b>p² = {p0**2:.4f}</b><br>
    ◈ Aa 이론 빈도: <b>2pq = {2*p0*q0:.4f}</b><br>
    ◻ aa 이론 빈도: <b>q² = {q0**2:.4f}</b>
  </p>
</div>

<div class="step-card" style="border-left-color:#10B981;">
  <h3>H-W 법칙이 성립하는 조건 (체크리스트)</h3>
  <p>
    {'✅' if w_AA == w_Aa == w_aa == 1.0 else '❌'} 자연선택 없음 (w_AA = w_Aa = w_aa = 1.0)<br>
    {'✅' if mu_Aa == 0 and mu_aA == 0 else '❌'} 돌연변이 없음 (μ = ν = 0)<br>
    {'✅' if not use_drift else '❌'} 무한 집단 (유전적 부동 없음)<br>
    ✅ 무작위 교배 (모델 기본 가정)
  </p>
</div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 실험 계획 요약")
        st.markdown(f"""
| 설정 항목 | 값 |
|---|---|
| 초기 p₀ | {p0:.2f} |
| 시뮬레이션 세대 | {generations}세대 |
| w(AA) / w(Aa) / w(aa) | {w_AA} / {w_Aa} / {w_aa} |
| 돌연변이율 μ / ν | {mu_Aa:.4f} / {mu_aA:.4f} |
| 유전적 부동 | {'활성화 (N=' + str(population_size) + ')' if use_drift else '비활성화 (무한 집단)'} |
        """)

        st.markdown("""
<div class="warn-box">
  💡 <b>탐구 팁</b>: 왼쪽 사이드바에서 한 번에 <b>하나의 변수만</b> 바꾸어 실험해야 인과관계를 파악할 수 있습니다.
</div>
        """, unsafe_allow_html=True)

    # 저장
    st.session_state["hypothesis"] = hypothesis
    st.session_state["research_question"] = research_question
    st.session_state["independent_var"] = independent_var
    st.session_state["dependent_var"] = dependent_var
    st.session_state["controlled_var"] = controlled_var


# ══════════════════════════════════════════
# 시뮬레이션 실행
# ══════════════════════════════════════════

if run_btn:
    config = SimulationConfig(
        p0=p0,
        w_AA=w_AA, w_Aa=w_Aa, w_aa=w_aa,
        mu_Aa=mu_Aa, mu_aA=mu_aA,
        population_size=population_size if use_drift else None,
        generations=generations,
    )
    with st.spinner("🔄 시뮬레이션 실행 중..."):
        df_hw, df_evo = run_experiment(config, rng_seed=int(rng_seed) if use_drift else None)
    st.session_state["df_hw"]  = df_hw
    st.session_state["df_evo"] = df_evo
    st.session_state["config"] = config
    st.success("✅ 시뮬레이션 완료! '시뮬레이션 결과' 탭으로 이동하세요.")


# ══════════════════════════════════════════
# TAB 2: 시뮬레이션 결과
# ══════════════════════════════════════════

with tab2:
    st.markdown("## 2️⃣ 시뮬레이션 결과")

    if st.session_state["df_hw"] is None:
        st.warning("⬅ 왼쪽 사이드바에서 설정 후 **시뮬레이션 실행** 버튼을 누르세요.")
    else:
        df_hw  = st.session_state["df_hw"]
        df_evo = st.session_state["df_evo"]

        # ── 결과 요약 메트릭
        st.markdown("### 📌 최종 세대 결과 요약")
        last_hw  = df_hw.iloc[-1]
        last_evo = df_evo.iloc[-1]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("p (H-W 이론)", f"{last_hw['p']:.4f}")
        c2.metric("p (진화 모델)", f"{last_evo['p']:.4f}",
                  delta=f"{last_evo['p'] - last_hw['p']:+.4f}")
        c3.metric("AA 오차", f"{abs(last_evo['freq_AA'] - last_hw['freq_AA']):.4f}")
        c4.metric("Aa 오차", f"{abs(last_evo['freq_Aa'] - last_hw['freq_Aa']):.4f}")
        c5.metric("aa 오차", f"{abs(last_evo['freq_aa'] - last_hw['freq_aa']):.4f}")

        # ── 그래프
        st.plotly_chart(plot_allele_frequency(df_hw, df_evo), use_container_width=True)
        st.plotly_chart(plot_genotype_frequency(df_hw, df_evo), use_container_width=True)
        st.plotly_chart(plot_mean_fitness(df_evo), use_container_width=True)

        # ── 데이터 테이블 (토글)
        with st.expander("🗃️ 원본 데이터 테이블 보기"):
            tab_hw_data, tab_evo_data = st.tabs(["H-W 이론 데이터", "진화 모델 데이터"])
            with tab_hw_data:
                st.dataframe(df_hw[["generation","p","q","freq_AA","freq_Aa","freq_aa"]].round(5),
                             use_container_width=True, height=300)
            with tab_evo_data:
                st.dataframe(df_evo[["generation","p","q","freq_AA","freq_Aa","freq_aa","mean_fitness"]].round(5),
                             use_container_width=True, height=300)


# ══════════════════════════════════════════
# TAB 3: 이론 vs 실제 비교
# ══════════════════════════════════════════

with tab3:
    st.markdown("## 3️⃣ H-W 이론 vs 진화 모델 비교")

    if st.session_state["df_hw"] is None:
        st.warning("⬅ 먼저 시뮬레이션을 실행하세요.")
    else:
        df_hw  = st.session_state["df_hw"]
        df_evo = st.session_state["df_evo"]

        st.markdown("### 모델이 언제, 얼마나 깨지는가?")
        st.info("""
        **탐구 포인트**: 아래 그래프에서 오차가 커지는 구간을 찾아보세요.
        - 어떤 요인이 H-W 평형을 가장 크게 깨뜨렸나요?
        - 어느 유전자형에서 오차가 가장 크게 나타났나요?
        """)

        st.plotly_chart(plot_hw_vs_evo_comparison(df_hw, df_evo), use_container_width=True)
        st.plotly_chart(plot_error(df_hw, df_evo), use_container_width=True)

        # ── 오차 통계 요약표
        st.markdown("### 📋 오차 통계 요약")
        merged = compute_deviation_summary(df_hw, df_evo)
        summary = merged[["err_p", "err_AA", "err_Aa", "err_aa"]].describe().T
        summary.index = ["p 오차", "AA 오차", "Aa 오차", "aa 오차"]
        summary.columns = ["개수", "평균", "표준편차", "최솟값", "25%", "중앙값", "75%", "최댓값"]
        st.dataframe(summary.round(5), use_container_width=True)

        # ── 탐구 질문
        st.markdown("---")
        st.markdown("### 🔍 탐구 질문 (토의)")
        q_col1, q_col2 = st.columns(2)
        with q_col1:
            st.markdown("""
**[질문 1]** 오차가 처음으로 0.01을 초과한 세대는 몇 세대인가요?

**[질문 2]** 자연선택의 방향성(AA 우세 vs aa 우세)이 오차 패턴에 어떤 영향을 주었나요?
            """)
        with q_col2:
            st.markdown("""
**[질문 3]** 집단 크기를 크게 바꾸면 오차 그래프는 어떻게 달라질까요? 실험해 보세요.

**[질문 4]** 돌연변이율을 0.001로 설정하면 p는 어떤 방향으로 수렴할까요?
            """)


# ══════════════════════════════════════════
# TAB 4: 결론 & 보고서
# ══════════════════════════════════════════

with tab4:
    st.markdown("## 4️⃣ 결론 도출 & 탐구 보고서")

    if st.session_state["df_hw"] is None:
        st.warning("⬅ 먼저 시뮬레이션을 실행하고 결과를 확인하세요.")
    else:
        df_hw  = st.session_state["df_hw"]
        df_evo = st.session_state["df_evo"]

        # 가설 다시 보여주기
        hypo = st.session_state.get("hypothesis", "")
        rq   = st.session_state.get("research_question", "")

        if hypo:
            st.markdown(f"""
<div class="hypothesis-box">
  <h4>📌 나의 가설 (재확인)</h4>
  <p>{hypo}</p>
</div>
            """, unsafe_allow_html=True)

        st.markdown("### ✍️ 결론 작성")
        st.caption("실험 데이터를 바탕으로 가설이 지지되었는지, 기각되었는지 서술하세요.")

        conclusion = st.text_area(
            "결론 (Conclusion)",
            placeholder=(
                "예: 실험 결과, AA의 적응도가 1.2로 높을 때 a 유전자의 빈도는 50세대 이후 0.15까지 감소하였다. "
                "이는 '자연선택 시 열성 대립유전자 빈도가 감소한다'는 가설을 지지한다. "
                "그러나 완전한 소멸은 일어나지 않았으며, 이는 잡형(Aa)이 a 유전자를 보존하기 때문으로 해석된다."
            ),
            height=160,
            key="conclusion",
        )

        discussion = st.text_area(
            "토의 (Discussion)",
            placeholder=(
                "예: H-W 평형이 성립하려면 자연선택이 없어야 하지만, 현실 집단에서는 이 조건이 항상 성립하지 않는다. "
                "돌연변이, 자연선택, 유전적 부동 중 어느 요인이 가장 큰 영향을 미쳤는지 비교해 보면 …"
            ),
            height=120,
            key="discussion",
        )

        # ── 데이터 기반 자동 요약
        st.markdown("### 📊 데이터 기반 실험 결과 요약 (자동 생성)")

        last_hw  = df_hw.iloc[-1]
        last_evo = df_evo.iloc[-1]
        cfg      = st.session_state["config"]
        delta_p  = last_evo["p"] - cfg.p0
        max_err  = compute_deviation_summary(df_hw, df_evo)["err_p"].max()

        st.markdown(f"""
| 항목 | 값 |
|---|---|
| 초기 p₀ | {cfg.p0:.3f} |
| 최종 p (H-W 이론) | {last_hw['p']:.4f} |
| 최종 p (진화 모델) | {last_evo['p']:.4f} |
| p 변화량 (Δp) | {delta_p:+.4f} |
| p 최대 오차 | {max_err:.4f} |
| 집단 평균 적응도 (최종) | {last_evo['mean_fitness']:.4f} |
        """)

        if abs(max_err) < 0.001:
            st.success("🟢 H-W 평형 유지: 오차가 매우 작아 이론값과 거의 일치합니다.")
        elif abs(max_err) < 0.05:
            st.warning("🟡 H-W 평형 약한 이탈: 일부 오차가 발생하였으나 크지 않습니다.")
        else:
            st.error("🔴 H-W 평형 붕괴: 진화 요인으로 인해 이론값과 큰 차이가 발생하였습니다.")

        # ── 보고서 다운로드
        st.markdown("---")
        st.markdown("### 💾 탐구 보고서 내보내기")

        report_text = f"""# 하디-바인베르크 & 진화 탐구 보고서

## 탐구 문제
{rq or '(미입력)'}

## 가설
{hypo or '(미입력)'}

## 실험 설계
- 초기 p₀ = {cfg.p0}
- 세대 수 = {cfg.generations}
- 적응도: w(AA)={cfg.w_AA}, w(Aa)={cfg.w_Aa}, w(aa)={cfg.w_aa}
- 돌연변이율: μ={cfg.mu_Aa}, ν={cfg.mu_aA}
- 유전적 부동: {'활성 (N=' + str(cfg.population_size) + ')' if cfg.population_size else '비활성'}

## 결과 요약
- 최종 p (H-W 이론): {last_hw['p']:.4f}
- 최종 p (진화 모델): {last_evo['p']:.4f}
- Δp = {delta_p:+.4f}
- 최대 오차: {max_err:.4f}

## 결론
{conclusion or '(미입력)'}

## 토의
{discussion or '(미입력)'}
"""

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "📄 보고서 텍스트 다운로드 (.txt)",
                data=report_text,
                file_name="hw_evolution_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_dl2:
            csv_data = compute_deviation_summary(df_hw, df_evo).to_csv(index=False)
            st.download_button(
                "📊 데이터 CSV 다운로드",
                data=csv_data,
                file_name="hw_evolution_data.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ─────────────────────────────────────────
# 하단 교사용 안내
# ─────────────────────────────────────────

with st.expander("👩‍🏫 교사용 수업 안내 (클릭하여 펼치기)", expanded=False):
    st.markdown("""
### 수업 활용 가이드

#### 수업 흐름 (70분 기준)
| 단계 | 시간 | 활동 |
|---|---|---|
| 도입 | 10분 | H-W 법칙 개념 설명, 가설 세우기 |
| 탐구 1 | 15분 | H-W 조건 확인 (모든 w=1, μ=0, 무한 집단) |
| 탐구 2 | 20분 | 자연선택 실험 (w 값 변경) |
| 탐구 3 | 10분 | 유전적 부동 실험 (N 변경) |
| 분석 | 10분 | 이론 vs 실제 비교 그래프 해석 |
| 정리 | 5분 | 결론 작성 및 공유 |

#### 권장 탐구 시나리오
1. **기준 실험**: w_AA=w_Aa=w_aa=1, μ=0, 무한 집단 → H-W 확인
2. **방향 선택**: w_AA=1.2, w_Aa=1.0, w_aa=0.8 → 우성 유전자 우위
3. **초열성**: w_AA=0.8, w_Aa=1.2, w_aa=0.8 → 잡형 유리 (다형성 유지)
4. **소집단**: N=30, 자연선택 없음 → 유전적 부동
5. **돌연변이 평형**: μ=0.001, ν=0.0001 → 평형 p 예측

#### 수학 교사를 위한 포인트
- `p² + 2pq + q² = 1` 이항전개와 연결
- 선택 후 p 갱신 공식: p' = (p²·w_AA + pq·w_Aa) / w̄ → 분수 함수 분석
- 오차 그래프를 절댓값 함수 개념과 연결

#### 생명과학 교사를 위한 포인트  
- 유전자풀, 대립유전자 빈도 개념과 연결
- 진화의 5가지 요인 (자연선택, 돌연변이, 부동, 유전자 흐름, 비무작위 교배)
- 모집단 유전학과 종분화의 연결고리
    """)
