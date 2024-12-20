import numpy as np

config = {
    "eta": 0.996,  # 結合損
    "alpha": 52.96,  # 伝搬損失係数
    "K": np.array([
        0.2619808632744556,
        0.02504964858993791,
        0.024556517817361553,
        0.02434552784608208,
        0.02095929830158411,
        0.06737091947446638,
        0.23497220134106522
    ]),  # 結合率
    "L": np.array(
        [
        6.2e-05,
        6.2e-05,
        6.2e-05,
        6.2e-05,
        7.749999999999999e-05,
        7.749999999999999e-05
    ]
    ),  # リング周長
    "n_eff": 2.2,  # 実行屈折率
    "n_g": 4.2,  # 群屈折率
    "center_wavelength": 1550e-9,
    "lambda_limit": np.arange(1510e-9, 1555e-9, 1e-12)
}
