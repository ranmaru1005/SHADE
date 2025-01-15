import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from MRR.evaluator import evaluate_band
from MRR.simulator import (
    calculate_practical_FSR,
    calculate_ring_length,
    calculate_x,
    optimize_N,
)
from MRR.transfer_function import simulate_transfer_function

@dataclass
class OptimizeKParams:
    L: npt.NDArray[np.float_]
    n_g: float
    n_eff: float
    eta: float
    alpha: float
    center_wavelength: float
    length_of_3db_band: float
    FSR: np.float_
    max_crosstalk: float
    H_p: float
    H_s: float
    H_i: float
    r_max: float
    weight: list[float]

def evaluate_with_error(
    K: npt.NDArray[np.float_],
    params: OptimizeKParams,
    error_value: float = 0.005
) -> None:
    """
    既存の結合率に誤差を加え、その評価値を再計算する。

    Parameters:
    - K: 理論的に最適化された結合率の配列
    - params: 最適化パラメータ
    - error_value: 結合率に加える誤差（デフォルトは 0.005）

    Output:
    - 各結合率の理論値、誤差を加えた値、評価値を表示
    """
    # 誤差を加えた結合率を生成
    perturbed_K = K + error_value
    perturbed_K = np.clip(perturbed_K, 0, params.eta)  # 範囲外はクリップ

    # 波長と透過特性の計算
    x = calculate_x(center_wavelength=params.center_wavelength, FSR=params.FSR)
    y_original = simulate_transfer_function(
        wavelength=x,
        L=params.L,
        K=K,
        alpha=params.alpha,
        eta=params.eta,
        n_eff=params.n_eff,
        n_g=params.n_g,
        center_wavelength=params.center_wavelength,
    )
    y_perturbed = simulate_transfer_function(
        wavelength=x,
        L=params.L,
        K=perturbed_K,
        alpha=params.alpha,
        eta=params.eta,
        n_eff=params.n_eff,
        n_g=params.n_g,
        center_wavelength=params.center_wavelength,
    )

    # 評価値の計算
    E_original = evaluate_band(
        x=x,
        y=y_original,
        center_wavelength=params.center_wavelength,
        length_of_3db_band=params.length_of_3db_band,
        max_crosstalk=params.max_crosstalk,
        H_p=params.H_p,
        H_s=params.H_s,
        H_i=params.H_i,
        r_max=params.r_max,
        weight=params.weight,
        ignore_binary_evaluation=False,
    )
    E_perturbed = evaluate_band(
        x=x,
        y=y_perturbed,
        center_wavelength=params.center_wavelength,
        length_of_3db_band=params.length_of_3db_band,
        max_crosstalk=params.max_crosstalk,
        H_p=params.H_p,
        H_s=params.H_s,
        H_i=params.H_i,
        r_max=params.r_max,
        weight=params.weight,
        ignore_binary_evaluation=False,
    )

    # 結果の出力
    print("結合率の評価結果:")
    for i in range(len(K)):
        print(f"K[{i}]: 理論値={K[i]:.5f}, 誤差値={perturbed_K[i]:.5f}")
    print(f"\n理論値の評価値: {E_original:.5f}")
    print(f"誤差を加えた評価値: {E_perturbed:.5f}")
    print(f"評価値の変動: {abs(E_original - E_perturbed):.5f}")

# 例としての結合率データとパラメータ
K_theoretical = np.array([0.54331948, 0.07410394, 0.02948469, 0.01517046, 0.11400472, 0.06921048, 0.43457259])  # 理論的な結合率
params = OptimizeKParams(
    L=np.array([7.75e-05, 6.20e-05, 6.20e-05, 7.75e-05, 7.75e-05,7.75e-05]),
    n_g=4.4,
    n_eff=2.2,
    eta=0.996,
    alpha=52.96,
    center_wavelength=1550e-9,
    length_of_3db_band=1e-9,
    FSR=35e-9,
    max_crosstalk=-30,
    H_p=-20,
    H_s=-60,
    H_i=-10,
    r_max=5,
    weight=[1.0, 3.5, 1.0, 5.0, 3.5, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
)

# 誤差を加えた評価値の計算
evaluate_with_error(K=K_theoretical, params=params)
