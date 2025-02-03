import argparse
import csv
import os
import subprocess
import datetime
from glob import glob
from importlib import import_module
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from jinja2 import Environment, PackageLoader

from config.model import SimulationConfig
from MRR.simulator import Accumulator, SimulatorResult, simulate_MRR




def plot_results(results, output_folder: Path, x_limits=None, y_limits=None):
    """シミュレーション結果をプロットし、元のグラフを保存"""
    
    for result in results:
        fig, ax = plt.subplots()

        # 🔹 NumPy 配列に変換（リストが来ても問題なし）
        x_data = np.array(result.x) * 1e9
        y_data = np.array(result.y)

        # 🔹 元のグラフをプロット
        ax.plot(x_data, y_data, label=result.label)
        ax.set_xlabel(r"Wavelength $\lambda$ (nm)", fontsize=14)
        ax.set_ylabel("Transmittance (dB)", fontsize=14)
        ax.set_title(f"Simulation Result: {result.name}")

        # 🔹 軸範囲の設定
        if x_limits:
            ax.set_xlim(x_limits)
        if y_limits:
            ax.set_ylim(y_limits)

        ax.legend()
        fig.savefig(output_folder / f"{result.name}_original.png")
        plt.close(fig)


def save_tsv_files(basedir: Path, results, x_limits=None, y_limits=None):
    """シミュレーション結果の tsv データを保存"""

    max_points = 2500
    steps = [(1 if np.array(result.x).size < max_points else np.array(result.x).size // max_points) for result in results]

    for result, step in zip(results, steps):
        # 🔹 NumPy 配列に変換
        x_data = np.array(result.x)
        y_data = np.array(result.y)

        # 🔹 全範囲のデータ保存
        with open(basedir / f"{result.name}_full.tsv", "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(x_data[::step], y_data[::step]))

        # 🔹 x 軸制限したデータ保存
        if x_limits:
            filtered_indices = (x_data >= x_limits[0]) & (x_data <= x_limits[1])
            x_filtered = x_data[filtered_indices]
            y_filtered = y_data[filtered_indices]

            if y_limits:
                filtered_indices = (y_filtered >= y_limits[0]) & (y_filtered <= y_limits[1])
                x_filtered = x_filtered[filtered_indices]
                y_filtered = y_filtered[filtered_indices]

            with open(basedir / f"{result.name}_filtered.tsv", "w") as tsvfile:
                tsv_writer = csv.writer(tsvfile, delimiter="\t")
                tsv_writer.writerows(zip(x_filtered, y_filtered))


if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument("NAME", help="from config.simulate import NAME", nargs="*")
    parser.add_argument("-l", "--list", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--format", action="store_true")
    parser.add_argument("-f", "--focus", action="store_true")
    parser.add_argument("-s", "--simulate-one-cycle", action="store_true")
    parser.add_argument("--x-min", type=float, default=1530, help="X-axis minimum value (nm)")
    parser.add_argument("--x-max", type=float, default=1570, help="X-axis maximum value (nm)")
    
    args = vars(parser.parse_args())
    ls = args["list"]
    skip_plot = args["skip_plot"]
    is_focus = args["focus"]
    format = args["format"]
    simulate_one_cycle = args["simulate_one_cycle"]

    # x軸の範囲を nm で指定
    x_limits = (args["x_min"], args["x_max"])

    results: list[SimulatorResult] = []
    accumulator = Accumulator(is_focus=is_focus)

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
                simulation_config.format = format
                simulation_config.simulate_one_cycle = simulate_one_cycle

                # 🔹 `lambda_limit` を x軸範囲に合わせて設定
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
                    format=simulation_config.format,
                    simulate_one_cycle=simulate_one_cycle,
                    lambda_limit=(x_limits[0] * 1e-9, x_limits[1] * 1e-9),  # nm → m に変換
                    name=simulation_config.name,
                    label=simulation_config.label,
                    skip_graph=False,
                    skip_evaluation=not simulate_one_cycle,
                )

                results.append(result)
                print("E:", result.evaluation_result)

            except ModuleNotFoundError as e:
                print(e)

        # 🔹 グラフ & TSV 保存処理を追加 🔹 #
        if not skip_plot:
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_folder = Path(f"graphs/{now}")
            output_folder.mkdir(parents=True, exist_ok=True)

            plot_results(results, output_folder, x_limits, y_limits=None)  # グラフを保存
            save_tsv_files(output_folder, results, x_limits)  # TSV を保存

            print(f"グラフと tsv を {output_folder} に保存しました。")
