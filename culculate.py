import numpy as np
from scipy.signal import argrelmin, argrelmax
from MRR.transfer_function import simulate_transfer_function
from MRR.simulator import calculate_x

# ---------------------------------------------------------------- #
# ▼▼▼ ハイブリッド版の評価関数群 ▼▼▼
# ---------------------------------------------------------------- #

def evaluate_graph_hybrid(x_nm, graph_db):
    """
    新しい3dB帯域幅計算と、元のリプル・クロストーク計算を組み合わせた評価。
    クロストークのロジックを正しく修正した最終版。
    """
    insertion_loss = evaluate_insertion_loss(graph_db)
    bandwidth_3db = calculate_bandwidth_nm(x_nm, graph_db, 3.0)
    
    # リプルは以前のハイブリッド版でOK
    ripple = evaluate_ripple_original_logic(graph_db)
    
    # ★クロストークを、今回修正した関数で計算
    crosstalk = evaluate_crosstalk_corrected(graph_db)

    print("--- Hybrid Evaluation Results (Corrected) ---")
    print(f"Insertion Loss = {insertion_loss:.3f} dB")
    print(f"3dB Bandwidth  = {bandwidth_3db:.3f} nm  <-- New, accurate method")
    print(f"Ripple         = {ripple:.3f} dB  <-- Original logic (Good)")
    print(f"Crosstalk      = {crosstalk:.3f} dB  <-- Original logic (Corrected)")
    return

# --- 帯域を正確に特定するための新しいヘルパー関数 ---
def _find_band_indices(y_transmittance, db_down):
    peak_value = np.max(y_transmittance)
    threshold = peak_value - db_down
    indices = np.where(y_transmittance >= threshold)[0]
    if len(indices) == 0:
        return None
    # 帯域の左端と右端のインデックスを返す
    return indices[0], indices[-1] 

# --- 新しい正確な帯域幅計算 ---
def calculate_bandwidth_nm(x_wavelength, y_transmittance, db_down):
    band_indices = _find_band_indices(y_transmittance, db_down)
    if band_indices is None:
        return 0.0
    left_idx, right_idx = band_indices
    return x_wavelength[right_idx] - x_wavelength[left_idx]

# --- 元々の評価関数（一部修正） ---
def evaluate_insertion_loss(graph_db):
    return np.max(graph_db)

def evaluate_ripple_original_logic(graph_db):
    # ★正確な帯域のインデックスを取得
    band_indices_3db = _find_band_indices(graph_db, 3.0)
    if band_indices_3db is None:
        return 0.0
    left_idx, right_idx = band_indices_3db
    
    # 元々のロジック：特定した帯域スライスに対して極大・極小を探す
    pass_band_slice = graph_db[left_idx : right_idx+1]
    point = _local_maximum_and_minimum(pass_band_slice)
    
    # point[1][0]は極大値のリスト, point[1][1]は極小値のリスト
    if not point[1][0] or not point[1][1]:
        return 0.0
    
    ripple = max(point[1][0]) - min(point[1][1])
    return ripple

def evaluate_crosstalk_original_logic(graph_db):
    # ★正確な帯域の右端インデックスを取得
    band_indices_3db = _find_band_indices(graph_db, 3.0)
    if band_indices_3db is None:
        return 999.0 # 評価不能
    _, right_idx = band_indices_3db

    # 元々のロジック：特定した帯域の右端から、最初の極大値を探す
    temp1 = 0
    temp2 = 0
    first_sidelobe_peak = -np.inf # マイナス無限大で初期化

    # グラフの端まで探索
    if right_idx + 1 >= len(graph_db):
        return 999.0 # 阻止域がない

    for i in range(right_idx + 1, len(graph_db)):
        temp1 = graph_db[i-1]
        temp2 = graph_db[i]
        if temp2 < temp1: # 最初のピーク（下り坂）を検出
            first_sidelobe_peak = temp1
            break
    
    if first_sidelobe_peak == -np.inf: # ピークが見つからなかった
        return 999.0

    return np.max(graph_db) - first_sidelobe_peak

# --- 元々のコードで使われていたヘルパー関数 ---
def _local_maximum_and_minimum(graph):
    max_point_y, min_point_y = [], []
    # 簡易的な実装（連続する2点間の比較）
    for i in range(1, len(graph) - 1):
        if graph[i-1] < graph[i] and graph[i] > graph[i+1]:
            max_point_y.append(graph[i])
        elif graph[i-1] > graph[i] and graph[i] < graph[i+1]:
            min_point_y.append(graph[i])
    # point_y[0]が極大値リスト, point_y[1]が極小値リスト
    return [[], [max_point_y, min_point_y]]


# ---------------------------------------------------------------- #
# ▼▼▼ メインの実行部分 ▼▼▼
# ---------------------------------------------------------------- #

x_m = calculate_x(center_wavelength=1550e-9, FSR=35e-9)
graph_db = simulate_transfer_function(
    wavelength=x_m,
    L=np.array([
        6.2e-05, 6.2e-05, 7.75e-05, 7.75e-05, 7.75e-05, 7.75e-05,
    ]),
    K=np.array([0.1019, 0.0255, 0.0240, 0.0397, 0.0859, 0.3755, 0.9292]),
    alpha=11.51, eta=0.996, n_eff=2.2, n_g=4.4, center_wavelength=1550e-9
)

x_nm = x_m * 1e9
# ハイブリッド版の評価関数を呼び出す
evaluate_graph_hybrid(x_nm, graph_db)

