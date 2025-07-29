
import argparse

from config.base import config
from MRR.logger import Logger
from MRR.model.DE import optimize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-plot", action="store_true")
    args = vars(parser.parse_args())
    skip_plot = args["skip_plot"]
    logger = Logger()
    logger.save_optimization_config(config)
    optimize(
        n_g=config.n_g,
        n_eff=config.n_eff,
        eta=config.eta,
        alpha=config.alpha,
        center_wavelength=config.center_wavelength,
        length_of_3db_band=config.length_of_3db_band,
        FSR=config.FSR,
        max_crosstalk=config.max_crosstalk,
        H_p=config.H_p,
        H_s=config.H_s,
        H_i=config.H_i,
        r_max=config.r_max,
        weight=config.weight,
        min_ring_length=config.min_ring_length,
        number_of_rings=config.number_of_rings,
        number_of_generations=config.number_of_generations,
        strategy=config.strategy,
        seedsequence=config.seedsequence,
        logger=logger,
        skip_plot=skip_plot,
    )
