import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="진화 가상 실험실",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .sidebar-section {
        font-size: 13px;
        font-weight: 600;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 16px 0 6px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #eee;
    }
    .hypothesis-box {
        background: #f0f6ff;
        border-left: 4px solid #378ADD;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 15px;
        color: #1a1a2e;
        line-height: 1.6;
    }
    .insight-box {
        background: #fff8e1;
        border-left: 4px solid #EF9F27;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 14px;
        color: #4a3800;
        line-height: 1.6;
    }
    h1 { margin-bottom: 0 !important; }
    .subtitle {
        color: #666;
        font-size: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


class EvolutionSimulator:
    def __init__(self, p_init, N, w_AA, w_Aa, w_aa, mu=0.0):
        self.p_init = float(p_init)
        self.N      = int(N)
        self.w_AA   = float(w_AA)
        self.w_Aa   = float(w_Aa)
        self.w_aa   = float(w_aa)
        self.mu     = float(mu)

    def run(self, generations):
        records = []
        curr_p = self.p_init

        hw_p  = self.p_init
        hw_q  = 1.0 - hw_p
        hw_AA = hw_p ** 2
        hw_Aa = 2.0 * hw_p * hw_q
        hw_aa = hw_q ** 2

        for gen in range(generations + 1):
            curr_q = 1.0 - curr_p
            actual_AA = curr_p ** 2
            actual_Aa = 2.0 * curr_p * curr_q
            actual_aa = curr_q ** 2
            error_p   = abs(hw_p - curr_p)

            records.append({
                "Generation": gen,
                "HW_p": hw_p,   "HW_q": hw_q,
                "HW_AA": hw_AA, "HW_Aa": hw_Aa, "HW_aa": hw_aa,
                "Evo_p": curr_p,     "Evo_q": curr_q,
                "Evo_AA": actual_AA, "Evo_Aa": actual_Aa, "Evo_aa": actual_aa,
                "Error_p": error_p,
            })

            if gen == generations:
                break

            w_bar = (curr_p**2)*self.w_AA + (2*curr_p*curr_q)*self.w_Aa + (curr_q**2)*self.w_aa
            p_sel = ((curr_p**2)*self.w_AA + curr_p*curr_q*self.w_Aa) / w_bar if w_bar > 0 else 0.0
            p_mut = float(np.clip(p_sel * (1.0 - self.mu), 0.0, 1.0))

            if self.N > 0:
                curr_p = np.random.binomial(2 * self.N, p_mut) / (2 * self.N)
            else:
                curr_p = p_mut

        return pd.DataFrame(records)


HW_COLOR  = "#888888"
EVO_COLOR = "#378ADD"
ERR_COLOR = "#E24B4A"
GENOTYPE_COLORS = {"AA": "#185FA5", "Aa": "#1D9E75", "aa": "#D85A30"}


def fig_allele(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Generation"], y=df["HW_p"],
        mode="lines", name="이론적 p (HWE)",
        line=dict(dash="dash", color=HW_COLOR, width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=df["Generation"], y=df["Evo_p"],
        mode="lines", name="실제 p (진화 시뮬레이션)",
        line=dict(color=EVO_COLOR, width=2.2),
    ))
    fig.update_layout(
        height=360, margin=dict(t=20, b=40, l=50, r=20),
        xaxis_title="세대 (Generation)", yaxis_title="대립유전자 빈도 p",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", y=1.08, x=0),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def fig_genotype(df, step=1):
    sampled = df[df["Generation"] % max(step, 1) == 0].copy()
    x_labels = []
    hw_AA, hw_Aa, hw_aa = [], [], []
    ev_AA, ev_Aa, ev_aa = [], [], []

    for _, row in sampled.iterrows():
        g = int(row["Generation"])
        x_labels += [f"{g}세대\n(이론)", f"{g}세대\n(실제)"]
        hw_AA.append(row["HW_AA"]); hw_AA.append(None)
        hw_Aa.append(row["HW_Aa"]); hw_Aa.append(None)
        hw_aa.append(row["HW_aa"]); hw_aa.append(None)
        ev_AA.append(None); ev_AA.append(row["Evo_AA"])
        ev_Aa.append(None); ev_Aa.append(row["Evo_Aa"])
        ev_aa.append(None); ev_aa.append(row["Evo_aa"])

    fig = go.Figure()
    for label, data, color, pat in [
        ("AA (이론)", hw_AA, GENOTYPE_COLORS["AA"], "/"),
        ("Aa (이론)", hw_Aa, GENOTYPE_COLORS["Aa"], "/"),
        ("aa (이론)", hw_aa, GENOTYPE_COLORS["aa"], "/"),
    ]:
        fig.add_trace(go.Bar(name=label, x=x_labels, y=data,
                             marker_color=color, marker_opacity=0.45,
                             marker_pattern_shape=pat))
    for label, data, color in [
        ("AA (실제)", ev_AA, GENOTYPE_COLORS["AA"]),
        ("Aa (실제)", ev_Aa, GENOTYPE_COLORS["Aa"]),
        ("aa (실제)", ev_aa, GENOTYPE_COLORS["aa"]),
    ]:
        fig.add_trace(go.Bar(name=label, x=x_labels, y=data, marker_color=color))

    fig.update_layout(
        barmode="stack", height=360, margin=dict(t=20, b=60, l=50, r=20),
        yaxis=dict(range=[0, 1], title="유전자형 빈도"), xaxis_title="세대",
        legend=dict(orientation="h", y=1.08, x=0, font_size=11),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def fig_error(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Generation"], y=df["Error_p"],
        mode="lines", name="|이론 p − 실제 p|",
        fill="tozeroy",
        line=dict(color=ERR_COLOR, width=2),
        fillcolor="rgba(226,75,74,0.12)",
    ))
    fig.update_layout(
        height=360, margin=dict(t=20, b=40, l=50, r=20),
        xaxis_title="세대 (Generation)", yaxis_title="오차 |Δp|",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", y=1.08, x=0),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


# ── 사이드바 ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 탐구 설계")

    st.markdown('<div class="sidebar-section">Step 1 · 과학적 가설 설정</div>', unsafe_allow_html=True)
    hypothesis = st.text_area(
        label="가설",
        placeholder="예: 집단 크기(N)가 작을수록 유전적 부동에 의해\n이론값(HWE)과의 오차가 커질 것이다.",
        height=100, label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section">Step 2 · 초기 조건 (통제변인)</div>', unsafe_allow_html=True)
    p_init      = st.slider("초기 A 대립유전자 빈도 (p₀)", 0.01, 0.99, 0.50, 0.01)
    generations = st.slider("관찰 세대 수", 10, 500, 100, 10)

    st.markdown('<div class="sidebar-section">Step 3 · 진화 요인 (조작변인)</div>', unsafe_allow_html=True)
    N = st.number_input("집단 크기 (N)", min_value=10, max_value=10_000, value=200, step=10)

    st.markdown("**생존가 (w)**")
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1: w_AA = st.number_input("AA", 0.0, 1.0, 1.0, 0.05)
    with col_w2: w_Aa = st.number_input("Aa", 0.0, 1.0, 1.0, 0.05)
    with col_w3: w_aa = st.number_input("aa", 0.0, 1.0, 1.0, 0.05)

    mu = st.slider("돌연변이율 μ (A→a)", 0.000, 0.050, 0.000, 0.001, format="%.3f")

    st.divider()
    run_btn = st.button("▶ 시뮬레이션 실행", use_container_width=True, type="primary")

    with st.expander("💡 탐구 시나리오 예시"):
        st.markdown("""
**① 대수의 법칙 & 유전적 부동**
- w 모두 1.0, μ = 0으로 고정
- N만 50 → 500 → 5000으로 바꿔 비교

**② 이형접합자 우위**
- w(AA)=0.8, w(Aa)=1.0, w(aa)=0.2
- p가 어디서 안정되는지 확인

**③ 돌연변이 압력**
- 생존가 모두 1.0, μ를 0.01 이상 설정
        """)


# ── 메인 ──────────────────────────────────────────────────
st.title("🧬 진화 가상 실험실")
st.markdown('<p class="subtitle">수학적 이상 모델(HWE)과 실제 진화 현상의 차이를 스스로 탐구합니다.</p>',
            unsafe_allow_html=True)

if hypothesis:
    st.markdown(f'<div class="hypothesis-box">📌 <strong>나의 가설:</strong> {hypothesis}</div>',
                unsafe_allow_html=True)
else:
    st.info("👈 사이드바에서 가설을 작성하고 시뮬레이션 실행 버튼을 누르세요.")

if run_btn or ("df" in st.session_state):
    if run_btn:
        sim = EvolutionSimulator(p_init, int(N), w_AA, w_Aa, w_aa, mu)
        df  = sim.run(generations)
        st.session_state["df"] = df
    else:
        df = st.session_state["df"]

    final_p   = df["Evo_p"].iloc[-1]
    avg_err   = df["Error_p"].mean()
    max_err   = df["Error_p"].max()
    final_err = df["Error_p"].iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("최종 실제 p",    f"{final_p:.3f}",  delta=f"{final_p - p_init:+.3f} vs 초기값")
    c2.metric("평균 오차",      f"{avg_err:.3f}")
    c3.metric("최대 오차",      f"{max_err:.3f}")
    c4.metric("최종 세대 오차", f"{final_err:.3f}")

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "📊 대립유전자 빈도 비교",
        "📈 유전자형 빈도 비교",
        "📉 오차 분석 (HWE 모델의 붕괴)",
    ])

    with tab1:
        st.markdown("**이론값(점선)은 항상 p₀에 고정됩니다. 실제값(실선)과의 차이를 관찰하세요.**")
        st.plotly_chart(fig_allele(df), use_container_width=True)

    with tab2:
        step = max(1, generations // 12)
        st.markdown(f"세대를 {step}세대 간격으로 추출하여 이론(빗금)과 실제(단색)를 비교합니다.")
        st.plotly_chart(fig_genotype(df, step=step), use_container_width=True)

    with tab3:
        st.plotly_chart(fig_error(df), use_container_width=True)

        if int(N) <= 50:
            q = "집단 크기가 매우 작아 유전적 부동이 강하게 작용합니다. p가 급격히 변동하는 패턴이 보이나요?"
        elif not (w_AA == 1.0 and w_Aa == 1.0 and w_aa == 1.0):
            q = "생존가 차이가 p의 방향을 결정합니다. 이형접합자(Aa) 우위라면 p는 어디서 평형을 이룰까요?"
        elif mu > 0:
            q = "돌연변이 압력이 p를 한 방향으로 밀고 있습니다. μ가 커질수록 속도는 어떻게 변할까요?"
        else:
            q = "오차가 0에서 멀어지는 패턴을 관찰하고, 설정한 조작변인과 연결하여 설명해 보세요."

        st.markdown(f'<div class="insight-box">💡 <strong>탐구 질문:</strong> {q}</div>',
                    unsafe_allow_html=True)

    st.divider()

    with st.expander("📋 시뮬레이션 데이터 보기 / 다운로드"):
        st.dataframe(df.round(5), use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ CSV 다운로드", data=csv,
                           file_name="evolution_simulation.csv", mime="text/csv")
