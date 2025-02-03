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


def plot_results(results: list[SimulatorResult], output_folder: Path, x_limits=None, y_limits=None) -> None:
    """シミュレーション結果をプロットし、元のグラフを保存"""
    
    for result in results:
        fig, ax = plt.subplots()

        # 1️⃣ 元のグラフ
        ax.plot(result.x * 1e9, result.y, label=result.label)  # nm単位に変換
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmittance (dB)")
        ax.set_ylim(-60, 0)  # y 軸範囲固定
        ax.set_xlim(x_limits)  # x 軸の範囲を適用
        ax.legend()
        fig.savefig(output_folder / f"{result.name}_original.png")
        plt.close(fig)


def save_tsv_files(basedir: Path, results: list[SimulatorResult], x_limits=None) -> None:
    """シミュレーション結果の tsv データを保存"""
    
    max_points = 2500
    steps = [(1 if result.x.size < max_points else result.x.size // max_points) for result in results]

    for result, step in zip(results, steps):
        # 🔹 全範囲のデータ保存
        with open(basedir / f"{result.name}_full.tsv", "w") as tsvfile:
            x = result.x[::step] * 1e9  # nm単位に変換
            y = result.y[::step]
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(x, y))

        # 🔹 x 軸制限したデータ保存
        filtered_indices = (result.x * 1e9 >= x_limits[0]) & (result.x * 1e9 <= x_limits[1])
        filtered_x = result.x[filtered_indices] * 1e9  # nm単位に変換
        filtered_y = result.y[filtered_indices]

        with open(basedir / f"{result.name}_filtered.tsv", "w") as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(filtered_x, filtered_y))


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
