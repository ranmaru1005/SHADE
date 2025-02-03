import argparse
import csv
import os
import subprocess
import datetime
from glob import glob
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
from jinja2 import Environment, PackageLoader

from config.model import SimulationConfig
from MRR.simulator import Accumulator, SimulatorResult, simulate_MRR


def plot_results(results: list[SimulatorResult], output_folder: Path) -> None:
    """シミュレーション結果をプロットし、グラフを保存"""
    
    for result in results:
        # 🔹 x 軸を 1530 nm ~ 1570 nm に制限
        filtered_indices = (result.x >= 1.53e-6) & (result.x <= 1.57e-6)
        x_filtered = result.x[filtered_indices] * 1e9  # nmに変換
        y_filtered = result.y[filtered_indices]

        fig, ax = plt.subplots()
        ax.plot(x_filtered, y_filtered, label=result.label)
        ax.set_xlabel("Wavelength λ (nm)")
        ax.set_ylabel("Transmittance (dB)")
        ax.set_title(f"Simulation Result: {result.name}")
        ax.set_xlim(1530, 1570)  # 🔹 x軸の範囲を 1530~1570 nm に設定
        ax.set_ylim(-60, 0)  # 🔹 y軸の範囲を -60 dB までに制限
        ax.tick_params(axis="x", direction="in")  # 目盛りを内向きに
        ax.tick_params(axis="y", direction="in")
        ax.legend()
        fig.savefig(output_folder / f"{result.name}_modified.png")
        plt.close(fig)


def save_tsv_files(basedir: Path, results: list[SimulatorResult]) -> None:
    """シミュレーション結果の tsv データを保存"""
    
    max_points = 2500
    steps = [(1 if result.x.size < max_points else result.x.size // max_points) for result in results]

    for result, step in zip(results, steps):
        # 🔹 x 軸を 1530 nm ~ 1570 nm に制限
        filtered_indices = (result.x >= 1.53e-6) & (result.x <= 1.57e-6)
        filtered_x = result.x[filtered_indices] * 1e9  # nmに変換
        filtered_y = result.y[filtered_indices]

        with open(basedir / f"{result.name}_filtered.tsv", "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(filtered_x, filtered_y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("NAME", help="from config.simulate import NAME", nargs="*")
    parser.add_argument("-l", "--list", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    
    args = vars(parser.parse_args())
    ls = args["list"]
    skip_plot = args["skip_plot"]

    results: list[SimulatorResult] = []
    accumulator = Accumulator()

    if ls:
        print("\t".join([os.path.splitext(os.path.basename(p))[0] for p in sorted(glob("config/simulate/*.py"))]))
    else:
        for name in args["NAME"]:
            if name.endswith(".py"):
                name = name[:-3]
            
            try:
                imported_module = import_module(f"config.simulate.{name}")
                imported_config = getattr(imported_module, "config")
                simulation_config = SimulationConfig(**imported_config)
                simulation_config.name = name

                result = simulate_MRR(
                    accumulator=accumulator,
                    L=simulation_config.L,
                    K=simulation_config.K,
                    n_eff=simulation_config.n_eff,
                    n_g=simulation_config.n_g,
                    eta=simulation_config.eta,
                    alpha=simulation_config.alpha,
                    center_wavelength=simulation_config.center_wavelength,
                    length_of_3db_band=simulation_config.length_of_3db_band,
                    max_crosstalk=simulation_config.max_crosstalk,
                    H_p=simulation_config.H_p,
                    H_i=simulation_config.H_i,
                    H_s=simulation_config.H_s,
                    r_max=simulation_config.r_max,
                    weight=simulation_config.weight,
                    name=simulation_config.name,
                    label=simulation_config.label,
                )

                results.append(result)
                print("E:", result.evaluation_result)

            except ModuleNotFoundError as e:
                print(e)

        if not skip_plot:
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_folder = Path(f"graphs/{now}")
            output_folder.mkdir(parents=True, exist_ok=True)

            plot_results(results, output_folder)  # グラフを保存
            save_tsv_files(output_folder, results)  # TSV を保存

            print(f"グラフと tsv を {output_folder} に保存しました。")
