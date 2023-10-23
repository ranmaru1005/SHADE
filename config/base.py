from config.model import OptimizationConfig

config = OptimizationConfig(
    eta=0.996,
    alpha=52.96,
    n_eff=2.2,
    n_g=4.4,
    number_of_rings=2,
    center_wavelength=1550e-9,
    FSR=20e-9,
    length_of_3db_band=1e-9,
    max_crosstalk=-30,
    H_p=-20,
    H_s=-60,
    H_i=-10,
    r_max=5,
    weight=[1.0, 3.5, 1.0, 5.0, 3.5, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    min_ring_length=50e-6,
    number_of_generations=300,
    strategy=[0.03, 0.07, 0.2, 0.7],
)
