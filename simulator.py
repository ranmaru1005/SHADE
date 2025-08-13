import argparse
import csv
import os
import datetime
from glob import glob
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # ▼▼▼ 変更点1: pandasをインポート ▼▼▼

from config.model import SimulationConfig
from MRR.simulator import Accumulator, SimulatorResult, simulate_MRR

# ... plot_combined_results と plot_results 関数は変更なし ...
def plot_combined_results(results: list[SimulatorResult], output_folder: Path, base_name: str, x_limits=None) -> None:
    """複数のシミュレーション結果を1つのグラフに重ねてプロットする"""
    fig, ax = plt.subplots()
    for result in results:
        ax.plot(result.x * 1e9, result.y, label=result.label)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmittance (dB)")
    ax.set_ylim(-60, 0)
    ax.legend()
    ax.tick_params(direction="in", length=6, width=1, which="both")
    fig.savefig(output_folder / f"{base_name}_original_combined.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    for result in results:
        ax.plot(result.x * 1e9, result.y, label=result.label)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmittance (dB)")
    ax.set_xlim(x_limits)
    ax.set_ylim(-60, 0)
    ax.legend()
    ax.tick_params(direction="in", length=6, width=1, which="both")
    fig.savefig(output_folder / f"{base_name}_modified_combined.png")
    plt.close(fig)

def plot_results(results: list[SimulatorResult], output_folder: Path, x_limits=None, y_limits=None) -> None:
    for result in results:
        fig, ax = plt.subplots()
        ax.plot(result.x * 1e9, result.y, label=result.label)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmittance (dB)")
        ax.set_ylim(-60, 0)
        ax.legend()
        ax.tick_params(direction="in", length=6, width=1, which="both")
        fig.savefig(output_folder / f"{result.name}_original.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(result.x * 1e9, result.y, label=result.label)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmittance (dB)")
        ax.set_xlim(x_limits)
        ax.set_ylim(-60, 0)
        ax.legend()
        ax.tick_params(direction="in", length=6, width=1, which="both")
        fig.savefig(output_folder / f"{result.name}_modified.png")
        plt.close(fig)


# ▼▼▼ 変更点2: TSV保存関数をExcel保存関数に置き換え ▼▼▼
def save_excel_file(basedir: Path, results: list[SimulatorResult], base_name: str, x_limits=None) -> None:
    """シミュレーション結果を1つのExcelファイルにシートを分けて保存"""
    
    excel_path = basedir / f"{base_name}.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        max_points = 2500 # TSVの時と同じく、多すぎる点は間引く
        
        for result in results:
            # --- 全範囲データのシート ---
            step = 1 if result.x.size < max_points else result.x.size // max_points
            df_full = pd.DataFrame({
                'Wavelength (nm)': result.x[::step] * 1e9,
                'Transmittance (dB)': result.y[::step]
            })
            # シート名にサニタイズが必要な文字が含まれる可能性を考慮
            safe_sheet_name_full = ''.join(c for c in result.label if c.isalnum() or c in (' ', '_'))[:25] + "_full"
            df_full.to_excel(writer, sheet_name=safe_sheet_name_full, index=False)

            # --- x軸制限したデータのシート ---
            filtered_indices = (result.x * 1e9 >= x_limits[0]) & (result.x * 1e9 <= x_limits[1])
            df_filtered = pd.DataFrame({
                'Wavelength (nm)': result.x[filtered_indices] * 1e9,
                'Transmittance (dB)': result.y[filtered_indices]
            })
            safe_sheet_name_filtered = ''.join(c for c in result.label if c.isalnum() or c in (' ', '_'))[:25] + "_filt"
            df_filtered.to_excel(writer, sheet_name=safe_sheet_name_filtered, index=False)


if __name__ == "__main__":
    # ... argparseの設定は変更なし ...
    parser = argparse.ArgumentParser()
    parser.add_argument("NAME", help="from config.simulate import NAME", nargs="*")
    parser.add_argument("-l", "--list", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--format", action="store_true")
    parser.add_argument("-f", "--focus", action="store_true")
    parser.add_argument("-s", "--simulate-one-cycle", action="store_true")
    parser.add_argument("--x-min", type=float, default=1530, help="X-axis minimum value (nm)")
    parser.add_argument("--x-max", type=float, default=1570, help="X-axis maximum value (nm)")
    parser.add_argument("--error-analysis", action="store_true", help="Run simulation with K +/- 0.005 errors and plot on one graph.")
    
    args = vars(parser.parse_args())
    
    ls = args["list"]
    skip_plot = args["skip_plot"]
    is_focus = args["focus"]
    format = args["format"]
    simulate_one_cycle = args["simulate_one_cycle"]
    x_limits = (args["x_min"], args["x_max"])
    error_analysis_mode = args["error_analysis"]

    if ls:
        print("\t".join([os.path.splitext(os.path.basename(p))[0] for p in sorted(glob("config/simulate/*.py"))]))
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_folder = Path(f"graphs/{now}")
        
        for name in args["NAME"]:
            if name.endswith(".py"): name = name[:-3]
            
            try:
                # ... configの読み込みは変更なし ...
                imported_module = import_module(f"config.simulate.{name}")
                imported_config = getattr(imported_module, "config")
                base_config = SimulationConfig(**imported_config)
                base_config.lambda_limit = (args["x_min"] * 1e-9, args["x_max"] * 1e-9)

                if error_analysis_mode:
                    print(f"--- Running Error Analysis for {name} ---")
                    results_for_analysis = []
                    error_val = 0.005
                    
                    k_original = np.array(base_config.K)
                    k_plus = np.clip(k_original + error_val, 0, 1)
                    k_minus = np.clip(k_original - error_val, 0, 1)
                    
                    k_configs = [
                        {"k": k_original, "label": "Original", "name": f"{name}_original"},
                        {"k": k_plus, "label": f"K + {error_val}", "name": f"{name}_plus_err"},
                        {"k": k_minus, "label": f"K - {error_val}", "name": f"{name}_minus_err"},
                    ]
                    
                    for k_config in k_configs:
                        print(f"Simulating with: {k_config['label']}")
                        result = simulate_MRR( K=k_config["k"], label=k_config["label"], name=k_config["name"], L=base_config.L, n_eff=base_config.n_eff, n_g=base_config.n_g, eta=base_config.eta, alpha=base_config.alpha, center_wavelength=base_config.center_wavelength, lambda_limit=base_config.lambda_limit, length_of_3db_band=base_config.length_of_3db_band, max_crosstalk=base_config.max_crosstalk, H_p=base_config.H_p, H_i=base_config.H_i, H_s=base_config.H_s, r_max=base_config.r_max, weight=base_config.weight, format=format, simulate_one_cycle=simulate_one_cycle, accumulator=Accumulator(is_focus=is_focus, init_graph=False), skip_graph=True, skip_evaluation=True, )
                        results_for_analysis.append(result)

                    if not skip_plot:
                        output_folder.mkdir(parents=True, exist_ok=True)
                        plot_combined_results(results_for_analysis, output_folder, base_name=name, x_limits=x_limits)
                        # ▼▼▼ 変更点3: TSV保存をExcel保存に変更 ▼▼▼
                        save_excel_file(output_folder, results_for_analysis, base_name=name, x_limits=x_limits)
                        print(f"Combined graph and Excel file saved to {output_folder}")

                else: # 通常モード
                    print(f"--- Running Standard Simulation for {name} ---")
                    result = simulate_MRR( K=base_config.K, label=name, name=name, L=base_config.L, n_eff=base_config.n_eff, n_g=base_config.n_g, eta=base_config.eta, alpha=base_config.alpha, center_wavelength=base_config.center_wavelength, lambda_limit=base_config.lambda_limit, length_of_3db_band=base_config.length_of_3db_band, max_crosstalk=base_config.max_crosstalk, H_p=base_config.H_p, H_i=base_config.H_i, H_s=base_config.H_s, r_max=base_config.r_max, weight=base_config.weight, format=format, simulate_one_cycle=simulate_one_cycle, accumulator=Accumulator(is_focus=is_focus, init_graph=False), skip_graph=True, skip_evaluation=not simulate_one_cycle, )
                    
                    if not skip_plot:
                        output_folder.mkdir(parents=True, exist_ok=True)
                        plot_results([result], output_folder, x_limits=x_limits)
                        # ▼▼▼ 変更点3: TSV保存をExcel保存に変更 ▼▼▼
                        save_excel_file(output_folder, [result], base_name=name, x_limits=x_limits)
                        print(f"Graph and Excel file saved to {output_folder}")

            except ModuleNotFoundError as e:
                print(e)
