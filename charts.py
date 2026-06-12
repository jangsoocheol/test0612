"""
charts.py
시뮬레이션 결과 시각화 모듈.
Plotly를 사용하여 인터랙티브 그래프를 생성한다.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────
# 디자인 토큰
# ─────────────────────────────────────────

PALETTE = {
    "hw":        "#6366F1",   # 보라 계열 - H-W 이론
    "evo":       "#F59E0B",   # 앰버 - 진화 모델
    "AA":        "#10B981",   # 에메랄드 - AA
    "Aa":        "#3B82F6",   # 블루 - Aa
    "aa":        "#EF4444",   # 레드 - aa
    "error":     "#EC4899",   # 핑크 - 오차
    "grid":      "rgba(200,200,200,0.3)",
    "bg":        "rgba(0,0,0,0)",
    "text":      "#1E293B",
}

FONT = dict(family="Noto Sans KR, sans-serif", size=13, color=PALETTE["text"])

BASE_LAYOUT = dict(
    font=FONT,
    plot_bgcolor=PALETTE["bg"],
    paper_bgcolor=PALETTE["bg"],
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(0,0,0,0.1)",
        borderwidth=1,
        font=dict(size=12),
    ),
    margin=dict(l=50, r=30, t=55, b=50),
    hovermode="x unified",
)


def _apply_base(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=15, color=PALETTE["text"])),
                      **BASE_LAYOUT)
    fig.update_xaxes(gridcolor=PALETTE["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=PALETTE["grid"], zeroline=False)
    return fig


# ─────────────────────────────────────────
# 그래프 1: p, q 대립유전자 빈도 변화
# ─────────────────────────────────────────

def plot_allele_frequency(df_hw: pd.DataFrame, df_evo: pd.DataFrame) -> go.Figure:
    """H-W 이론 p(일정)와 진화 모델 p 변화를 비교한다."""
    fig = go.Figure()

    # H-W 이론 (점선)
    fig.add_trace(go.Scatter(
        x=df_hw["generation"], y=df_hw["p"],
        mode="lines", name="p (H-W 이론)",
        line=dict(color=PALETTE["hw"], dash="dash", width=2),
        hovertemplate="세대 %{x}: p = %{y:.4f}<extra>H-W 이론</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_hw["generation"], y=1 - df_hw["p"],
        mode="lines", name="q (H-W 이론)",
        opacity=0.6,
        line=dict(color=PALETTE["hw"], dash="dot", width=1.5),
        hovertemplate="세대 %{x}: q = %{y:.4f}<extra>H-W 이론</extra>",
    ))

    # 진화 모델 (실선)
    fig.add_trace(go.Scatter(
        x=df_evo["generation"], y=df_evo["p"],
        mode="lines", name="p (진화 모델)",
        line=dict(color=PALETTE["evo"], width=2.5),
        hovertemplate="세대 %{x}: p = %{y:.4f}<extra>진화 모델</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_evo["generation"], y=df_evo["q"],
        mode="lines", name="q (진화 모델)",
        opacity=0.7,
        line=dict(color=PALETTE["evo"], dash="dot", width=1.5),
        hovertemplate="세대 %{x}: q = %{y:.4f}<extra>진화 모델</extra>",
    ))

    fig.update_yaxes(range=[-0.02, 1.02], title_text="대립유전자 빈도")
    fig.update_xaxes(title_text="세대 (generation)")
    return _apply_base(fig, "📈 대립유전자 빈도 변화: H-W 이론 vs 진화 모델")


# ─────────────────────────────────────────
# 그래프 2: 유전자형 빈도 변화 (진화 모델)
# ─────────────────────────────────────────

def plot_genotype_frequency(df_hw: pd.DataFrame, df_evo: pd.DataFrame) -> go.Figure:
    """유전자형(AA, Aa, aa) 빈도를 이론값과 실제값으로 나란히 비교한다."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("H-W 이론 유전자형 빈도", "진화 모델 유전자형 빈도"),
        shared_yaxes=True,
    )

    for col, (df, label) in enumerate([(df_hw, "이론"), (df_evo, "실제")], start=1):
        fig.add_trace(go.Scatter(
            x=df["generation"], y=df["freq_AA"],
            name=f"AA ({label})",
            mode="lines",
            line=dict(color=PALETTE["AA"], width=2, dash="dash" if col == 1 else "solid"),
            showlegend=(col == 2),
            hovertemplate=f"세대 %{{x}}: AA = %{{y:.4f}}<extra>AA {label}</extra>",
        ), row=1, col=col)

        fig.add_trace(go.Scatter(
            x=df["generation"], y=df["freq_Aa"],
            name=f"Aa ({label})",
            mode="lines",
            line=dict(color=PALETTE["Aa"], width=2, dash="dash" if col == 1 else "solid"),
            showlegend=(col == 2),
            hovertemplate=f"세대 %{{x}}: Aa = %{{y:.4f}}<extra>Aa {label}</extra>",
        ), row=1, col=col)

        fig.add_trace(go.Scatter(
            x=df["generation"], y=df["freq_aa"],
            name=f"aa ({label})",
            mode="lines",
            line=dict(color=PALETTE["aa"], width=2, dash="dash" if col == 1 else "solid"),
            showlegend=(col == 2),
            hovertemplate=f"세대 %{{x}}: aa = %{{y:.4f}}<extra>aa {label}</extra>",
        ), row=1, col=col)

    fig.update_yaxes(range=[-0.02, 1.02], title_text="유전자형 빈도", col=1)
    fig.update_xaxes(title_text="세대", row=1, col=1)
    fig.update_xaxes(title_text="세대", row=1, col=2)
    _apply_base(fig, "🧬 유전자형 빈도: H-W 이론(점선) vs 진화 모델(실선)")
    return fig


# ─────────────────────────────────────────
# 그래프 3: 이론값 vs 실제값 직접 비교
# ─────────────────────────────────────────

def plot_hw_vs_evo_comparison(df_hw: pd.DataFrame, df_evo: pd.DataFrame) -> go.Figure:
    """특정 세대의 유전자형 빈도를 막대 그래프로 나란히 비교한다."""
    genotypes = ["AA", "Aa", "aa"]
    colors = [PALETTE["AA"], PALETTE["Aa"], PALETTE["aa"]]

    # 최종 세대 기준
    last_hw = df_hw.iloc[-1]
    last_evo = df_evo.iloc[-1]

    hw_vals  = [last_hw["freq_AA"],  last_hw["freq_Aa"],  last_hw["freq_aa"]]
    evo_vals = [last_evo["freq_AA"], last_evo["freq_Aa"], last_evo["freq_aa"]]

    fig = go.Figure()
    for i, (gt, color) in enumerate(zip(genotypes, colors)):
        fig.add_trace(go.Bar(
            name=f"{gt} (H-W 이론)",
            x=[gt], y=[hw_vals[i]],
            marker_color=color, opacity=0.45,
            marker_line=dict(color=color, width=2),
            hovertemplate=f"{gt} H-W 이론: %{{y:.4f}}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name=f"{gt} (진화 모델)",
            x=[gt], y=[evo_vals[i]],
            marker_color=color, opacity=0.9,
            hovertemplate=f"{gt} 진화 모델: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_yaxes(range=[0, 1.05], title_text="빈도")
    fig.update_xaxes(title_text="유전자형")
    return _apply_base(fig, f"📊 최종 세대({int(last_evo['generation'])}세대) 유전자형 빈도 비교")


# ─────────────────────────────────────────
# 그래프 4: 오차 그래프
# ─────────────────────────────────────────

def plot_error(df_hw: pd.DataFrame, df_evo: pd.DataFrame) -> go.Figure:
    """세대별 |이론 - 실제| 오차를 시각화한다."""
    from simulation import compute_deviation_summary
    merged = compute_deviation_summary(df_hw, df_evo)

    fig = go.Figure()

    fill_colors = {
        "err_p":  "rgba(245,158,11,0.07)",
        "err_AA": "rgba(16,185,129,0.07)",
        "err_Aa": "rgba(59,130,246,0.07)",
        "err_aa": "rgba(239,68,68,0.07)",
    }
    for col, name, color in [
        ("err_p",  "p 빈도 오차",  PALETTE["evo"]),
        ("err_AA", "AA 오차",       PALETTE["AA"]),
        ("err_Aa", "Aa 오차",       PALETTE["Aa"]),
        ("err_aa", "aa 오차",       PALETTE["aa"]),
    ]:
        fig.add_trace(go.Scatter(
            x=merged["generation"], y=merged[col],
            mode="lines", name=name,
            line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=fill_colors[col],
            hovertemplate=f"세대 %{{x}}: {name} = %{{y:.4f}}<extra></extra>",
        ))

    fig.update_yaxes(title_text="|이론값 − 실제값|", rangemode="tozero")
    fig.update_xaxes(title_text="세대 (generation)")
    return _apply_base(fig, "⚠️ H-W 이론과 진화 모델의 오차 (|이론 − 실제|)")


# ─────────────────────────────────────────
# 그래프 5: 평균 적응도 변화
# ─────────────────────────────────────────

def plot_mean_fitness(df_evo: pd.DataFrame) -> go.Figure:
    """진화 모델에서 집단의 평균 적응도 변화를 나타낸다."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_evo["generation"], y=df_evo["mean_fitness"],
        mode="lines+markers",
        name="평균 적응도 (w̄)",
        line=dict(color=PALETTE["evo"], width=2.5),
        marker=dict(size=4),
        hovertemplate="세대 %{x}: w̄ = %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color=PALETTE["hw"],
                  annotation_text="H-W 기준 (w̄=1)", annotation_position="top right")
    fig.update_yaxes(title_text="평균 적응도 (w̄)")
    fig.update_xaxes(title_text="세대 (generation)")
    return _apply_base(fig, "💪 집단 평균 적응도 변화")
