import numpy as np

config = {
    "eta": 0.996,  # 結合損
    "alpha": 52.96,  # 伝搬損失係数
    "K": np.array([
        0.10689,
        0.04268,
        0.05098,
        0.05410,
        0.07308,
        0.15398,
        0.68369
    ]),  # 結合率
    "L": np.array(
        [
        6.2e-05,
        6.2e-05,
        7.749999999999999e-05,
        7.749999999999999e-05,
        7.749999999999999e-05,
        7.749999999999999e-05
    ]
    ),  # リング周長
    "n_eff": 2.2,  # 実行屈折率
    "n_g": 4.2,  # 群屈折率
    "center_wavelength": 1550e-9,
    "lambda_limit": np.arange(1510e-9, 1555e-9, 1e-12)
}
