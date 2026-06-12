"""
simulation.py
하디-바인베르크 이론 모델과 진화 모델을 분리하여 구현한 시뮬레이션 엔진.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────

@dataclass
class GenerationRecord:
    """한 세대의 모든 상태를 담는 레코드."""
    generation: int
    p: float          # A 대립유전자 빈도
    q: float          # a 대립유전자 빈도
    freq_AA: float    # AA 유전자형 빈도
    freq_Aa: float    # Aa 유전자형 빈도
    freq_aa: float    # aa 유전자형 빈도
    # 하디-바인베르크 이론값
    hw_AA: float = 0.0
    hw_Aa: float = 0.0
    hw_aa: float = 0.0
    # 오차 (|실제 - 이론|)
    error_AA: float = 0.0
    error_Aa: float = 0.0
    error_aa: float = 0.0
    mean_fitness: float = 1.0


@dataclass
class SimulationConfig:
    """시뮬레이션 설정값 묶음."""
    # 초기 대립유전자 빈도
    p0: float = 0.5

    # 적응도 (fitness)
    w_AA: float = 1.0
    w_Aa: float = 1.0
    w_aa: float = 1.0

    # 돌연변이율 (A → a, a → A)
    mu_Aa: float = 0.0    # A → a
    mu_aA: float = 0.0    # a → A

    # 집단 크기 (유전적 부동용; None이면 무한대 → 부동 없음)
    population_size: Optional[int] = None

    # 세대 수
    generations: int = 50


# ─────────────────────────────────────────
# 하디-바인베르크 이론 모델 (변화 없는 기준선)
# ─────────────────────────────────────────

class HardyWeinbergModel:
    """
    H-W 평형 조건에서의 이론적 유전자형 빈도를 계산한다.
    - 자연선택 없음 (w = 1)
    - 돌연변이 없음
    - 무한 집단 (부동 없음)
    - 무작위 교배
    """

    def run(self, p0: float, generations: int) -> list[GenerationRecord]:
        """초기 p0에서 generations 세대 동안 H-W 이론값을 계산한다."""
        records = []
        p = p0
        q = 1.0 - p

        for gen in range(generations + 1):
            hw_AA = p ** 2
            hw_Aa = 2 * p * q
            hw_aa = q ** 2

            records.append(GenerationRecord(
                generation=gen,
                p=p, q=q,
                freq_AA=hw_AA,
                freq_Aa=hw_Aa,
                freq_aa=hw_aa,
                hw_AA=hw_AA,
                hw_Aa=hw_Aa,
                hw_aa=hw_aa,
                error_AA=0.0, error_Aa=0.0, error_aa=0.0,
                mean_fitness=1.0,
            ))
            # H-W 하에서 p는 변하지 않음 → 루프 유지는 시각화 편의용

        return records

    @staticmethod
    def predict(p: float) -> tuple[float, float, float]:
        """주어진 p에서 H-W 예측값 (AA, Aa, aa) 반환."""
        q = 1.0 - p
        return p ** 2, 2 * p * q, q ** 2


# ─────────────────────────────────────────
# 진화 모델 (자연선택 + 돌연변이 + 부동)
# ─────────────────────────────────────────

class EvolutionModel:
    """
    실제 진화 요인을 반영한 집단유전학 시뮬레이션.

    알고리즘 순서 (매 세대):
      1. H-W 기대 유전자형 빈도 계산 (교배)
      2. 자연선택 적용 → 선택 후 p, q 갱신
      3. 돌연변이 적용 → p, q 재갱신
      4. 유전적 부동 (유한 집단 샘플링) → p, q 최종 갱신
      5. 이론값 비교 및 오차 계산
    """

    def __init__(self, config: SimulationConfig, rng_seed: Optional[int] = None):
        self.cfg = config
        self.rng = np.random.default_rng(rng_seed)

    def run(self) -> list[GenerationRecord]:
        cfg = self.cfg
        p = cfg.p0
        q = 1.0 - p
        records = []

        for gen in range(cfg.generations + 1):
            # ── 1. 현재 빈도에서 H-W 유전자형 빈도 계산
            freq_AA = p ** 2
            freq_Aa = 2 * p * q
            freq_aa = q ** 2

            # ── 2. H-W 이론값 (선택 전 p 기준)
            hw_AA, hw_Aa, hw_aa = freq_AA, freq_Aa, freq_aa

            # ── 3. 오차
            err_AA = abs(freq_AA - hw_AA)
            err_Aa = abs(freq_Aa - hw_Aa)
            err_aa = abs(freq_aa - hw_aa)

            # ── 4. 평균 적응도
            mean_w = (
                freq_AA * cfg.w_AA
                + freq_Aa * cfg.w_Aa
                + freq_aa * cfg.w_aa
            )

            records.append(GenerationRecord(
                generation=gen,
                p=p, q=q,
                freq_AA=freq_AA,
                freq_Aa=freq_Aa,
                freq_aa=freq_aa,
                hw_AA=hw_AA,
                hw_Aa=hw_Aa,
                hw_aa=hw_aa,
                error_AA=err_AA,
                error_Aa=err_Aa,
                error_aa=err_aa,
                mean_fitness=mean_w,
            ))

            if gen == cfg.generations:
                break

            # ────────────────────────────────
            # 자연선택 (p 갱신)
            # ────────────────────────────────
            if mean_w > 0:
                p_new = (freq_AA * cfg.w_AA + 0.5 * freq_Aa * cfg.w_Aa) / mean_w
            else:
                p_new = p
            p = np.clip(p_new, 0.0, 1.0)

            # ────────────────────────────────
            # 돌연변이
            # ────────────────────────────────
            # A → a : p를 mu_Aa 비율로 감소
            # a → A : q를 mu_aA 비율로 감소 → p 증가
            p = p * (1 - cfg.mu_Aa) + (1 - p) * cfg.mu_aA
            p = np.clip(p, 0.0, 1.0)

            # ────────────────────────────────
            # 유전적 부동 (이항 샘플링)
            # ────────────────────────────────
            if cfg.population_size is not None and cfg.population_size > 0:
                # 2N 개의 대립유전자 중 A가 binomial(2N, p)개
                alleles_A = self.rng.binomial(
                    2 * cfg.population_size, p
                )
                p = alleles_A / (2 * cfg.population_size)
                p = np.clip(p, 0.0, 1.0)

            q = 1.0 - p

            # 대립유전자 고정 확인 (더 이상 변화 없음)
            if p <= 0.0 or p >= 1.0:
                # 남은 세대를 고정 상태로 채움
                fixed_p = max(0.0, min(1.0, p))
                for remaining_gen in range(gen + 1, cfg.generations + 1):
                    fq = 1.0 - fixed_p
                    hw_AA = fixed_p ** 2
                    hw_Aa = 2 * fixed_p * fq
                    hw_aa = fq ** 2
                    records.append(GenerationRecord(
                        generation=remaining_gen,
                        p=fixed_p, q=fq,
                        freq_AA=hw_AA,
                        freq_Aa=hw_Aa,
                        freq_aa=hw_aa,
                        hw_AA=hw_AA,
                        hw_Aa=hw_Aa,
                        hw_aa=hw_aa,
                        mean_fitness=1.0,
                    ))
                break

        return records


# ─────────────────────────────────────────
# 유틸: 레코드 리스트 → DataFrame
# ─────────────────────────────────────────

def records_to_df(records: list[GenerationRecord]) -> pd.DataFrame:
    return pd.DataFrame([vars(r) for r in records])


# ─────────────────────────────────────────
# 통합 실험 실행 함수
# ─────────────────────────────────────────

def run_experiment(
    config: SimulationConfig,
    rng_seed: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    H-W 이론 모델과 진화 모델을 함께 실행하여
    두 DataFrame을 반환한다.

    Returns
    -------
    df_hw   : H-W 이론값 DataFrame
    df_evo  : 진화 모델 결과 DataFrame
    """
    hw_model = HardyWeinbergModel()
    df_hw = records_to_df(hw_model.run(config.p0, config.generations))

    evo_model = EvolutionModel(config, rng_seed=rng_seed)
    df_evo = records_to_df(evo_model.run())

    # H-W 이론값 열을 진화 모델 DataFrame에 병합하여 비교 편의 제공
    # (진화 df에는 이미 hw_* 컬럼이 있으므로 그대로 사용)
    return df_hw, df_evo


# ─────────────────────────────────────────
# 오차 요약 통계
# ─────────────────────────────────────────

def compute_deviation_summary(df_hw: pd.DataFrame, df_evo: pd.DataFrame) -> pd.DataFrame:
    """세대별 H-W 이론값과 진화 모델 실제값 사이의 오차를 계산한다."""
    merged = df_hw[["generation", "freq_AA", "freq_Aa", "freq_aa", "p"]].rename(
        columns={"freq_AA": "hw_AA", "freq_Aa": "hw_Aa",
                 "freq_aa": "hw_aa", "p": "hw_p"}
    ).merge(
        df_evo[["generation", "freq_AA", "freq_Aa", "freq_aa", "p"]].rename(
            columns={"freq_AA": "evo_AA", "freq_Aa": "evo_Aa",
                     "freq_aa": "evo_aa", "p": "evo_p"}
        ),
        on="generation"
    )
    merged["err_AA"] = (merged["evo_AA"] - merged["hw_AA"]).abs()
    merged["err_Aa"] = (merged["evo_Aa"] - merged["hw_Aa"]).abs()
    merged["err_aa"] = (merged["evo_aa"] - merged["hw_aa"]).abs()
    merged["err_p"]  = (merged["evo_p"]  - merged["hw_p"]).abs()
    return merged
