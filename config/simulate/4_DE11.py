import numpy as np

config = {
    "eta": 1,  # 結合損
    "alpha": 52.96,  # 伝搬損失係数
    "K": np.array([0.39623247, 0.10733708, 0.0874449, 0.10801439, 0.39519958]),  # 結合率
    "L": np.array([8.49490247e-05, 1.13265366e-04, 8.49490247e-05, 1.13265366e-04]),  # リング周長
    "n_eff": 3.3938,  # 実行屈折率
    "n_g": 4.2,  # 群屈折率
    "center_wavelength": 1550e-9,
    "lambda_limit": np.arange(1525e-9, 1575e-9, 1e-12),
}