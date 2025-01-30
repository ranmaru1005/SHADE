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


def plot_with_pgfplots(basedir: Path, results: list[SimulatorResult], is_focus: bool) -> None:
    """pgfplots を用いたグラフを作成し、保存"""
    max_points = 2500
    steps = [(1 if result.x.size < max_points else result.x.size // max_points) for result in results]
    
    # tsv ファイルを保存
    for result, step in zip(results, steps):
        with open(f"{basedir}/{result.name}_pgfplots.tsv", "w") as tsvfile:
            x = result.x[::step]
            y = result.y[::step]
            tsv_writer = csv.writer(tsvfile, delimiter="\t")
            tsv_writer.writerows(zip(x, y))

    # LaTeX テンプレートを使ってプロット作成
    env = Environment(loader=PackageLoader("MRR"))
    template = env.get_template("pgfplots.tex.j2")
    legends = "{" + ",".join([result.label for result in results]) + "}"
    tsvnames = ["{" + result.name + "_pgfplots.tsv}" for result in results]
    
    with open(basedir / "pgfplots.tex", "w") as fp:
        fp.write(template.render(tsvnames=tsvnames, legends=legends, is_focus=is_focus))
    
    subprocess.run(["lualatex", "pgfplots"], cwd=basedir, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument("NAME", help="from config.simulate import NAME", nargs="*")
    parser.add_argument("-l", "--list", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--format", action="store_true")
    parser.add_argument("-f", "--focus", action="store_true")
    parser.add_argument("-s", "--simulate-one-cycle", action="store_true")
    
    args = vars(parser.parse_args())
    ls = args["list"]
    skip_plot = args["skip_plot"]
    is_focus = args["focus"]
    format = args["format"]
    simulate_one_cycle = args["simulate_one_cycle"]

    results: list[SimulatorResult] = []
    accumulator = Accumulator(is_focus=is_focus)

    # シミュレーション設定リストの表示
    if ls:
        print("\t".join([os.path.splitext(os.path.basename(p))[0] for p in sorted(glob("config/simulate/*.py"))]))
    else:
        for name in args["NAME"]:
            if name.endswith(".py"):
                name = name[:-3]
            
            try:
                # 設定ファイルのインポート
                imported_module = import_module(f"config.simulate.{name}")
                imported_config = getattr(imported_module, "config")
                simulation_config = SimulationConfig(**imported_config)
                simulation_config.name = name
                simulation_config.format = format
                simulation_config.simulate_one_cycle = simulate_one_cycle

                # シミュレーション実行
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
                    lambda_limit=simulation_config.lambda_limit,
                    name=simulation_config.name,
                    label=simulation_config.label,
                    skip_graph=False,
                    skip_evaluation=not simulate_one_cycle,
                )

                results.append(result)
                print("E:", result.evaluation_result)

            except ModuleNotFoundError as e:
                print(e)

        #  グラフ保存処理を追加  #
        if not skip_plot:
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_folder = Path(f"graphs/{now}")
            output_folder.mkdir(parents=True, exist_ok=True)

            # 1️ 元のグラフ
            fig1 = accumulator.plot()
            fig1.savefig(output_folder / "original_plot.png")
            plt.close(fig1)

            # 2️ x 軸の範囲を変更したグラフ
            fig2 = accumulator.plot()
            plt.xlim(1.50, 1.60)  # x 軸を 1.50 ～ 1.60 に変更
            fig2.savefig(output_folder / "modified_x_range_plot.png")
            plt.close(fig2)

            print(f"グラフを {output_folder} に保存しました。")

            accumulator.show()  # 画面に表示も可能
