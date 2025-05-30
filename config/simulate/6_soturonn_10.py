import numpy as np

config = {
    "eta": 0.996,  # 結合損
    "alpha": 52.96,  # 伝搬損失係数
    "K": np.array([
        0.19557107762324696,
        0.05303311939308467,
        0.025273549963166223,
        0.024876160067680998,
        0.023656903093376275,
        0.028112032816433295,
        0.3218694099339701
    ]),  # 結合率
    "L": np.array(
        [
        7.75e-05,
        7.75e-05,
        6.2e-05,
        7.75e-05,
        6.2e-05,
        6.2e-05
    ]
    ),  # リング周長
    "n_eff": 2.2,  # 実行屈折率
    "n_g": 4.2,  # 群屈折率
    "center_wavelength": 1550e-9,
    "lambda_limit": np.arange(1510e-9, 1555e-9, 1e-12)
}
