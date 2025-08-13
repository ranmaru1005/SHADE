import numpy as np
from MRR.simulator import (
    calculate_practical_FSR,
    calculate_ring_length,
    calculate_x,
    optimize_N,
)
from MRR.transfer_function import simulate_transfer_function


#もともとのやつ
"""
def evaluate_graph(graph):                                  #まとめて出力するためのもの
    evaluate_data = [evaluate_insertion_loss(graph),evaluate_3dbband(graph)*0.01,evaluate_ripple(graph),evaluate_crosstalk(graph),evaluate_shape_factor(graph)]
    print("insertion loss = {} dB".format(evaluate_data[0]))
    print("3dbband = {} nm".format(evaluate_data[1]))
    print("ripple = {} dB".format(evaluate_data[2]))
    print("crosstalk = {} dB".format(evaluate_data[3]))
    print("shape_factor = {}".format(evaluate_data[4]))
    return
"""


def evaluate_graph_improved(x_nm, graph_db):
    #------------------------------------------------------
    #改善された評価関数群をまとめて呼び出し、結果を表示する。
    #引数にはx軸の波長データ(nm単位)も必要。
    #------------------------------------------------------
    insertion_loss = evaluate_insertion_loss(graph_db)
    bandwidth_3db = calculate_bandwidth_nm(x_nm, graph_db, 3.0)
    ripple = evaluate_ripple_improved(graph_db)
    crosstalk = evaluate_crosstalk_improved(graph_db)
    
    # 形状係数の計算（1dB帯域幅 / 10dB帯域幅）
    bandwidth_1db = calculate_bandwidth_nm(x_nm, graph_db, 1.0)
    bandwidth_10db = calculate_bandwidth_nm(x_nm, graph_db, 10.0)
    shape_factor = bandwidth_1db / bandwidth_10db if bandwidth_10db != 0 else 0

    print("--- Improved Evaluation Results ---")
    print(f"Insertion Loss = {insertion_loss:.3f} dB")
    print(f"3dB Bandwidth  = {bandwidth_3db:.3f} nm")
    print(f"Ripple         = {ripple:.3f} dB")
    print(f"Crosstalk      = {crosstalk:.3f} dB")
    print(f"Shape Factor   = {shape_factor:.3f}")
    return

def _find_band_indices(y_transmittance, db_down):
    """
    スペクトルデータから、ピークより指定dBだけ低いレベルの
    内側にある全てのデータ点のインデックスを返すヘルパー関数。
    """
    peak_value = np.max(y_transmittance)
    threshold = peak_value - db_down
    
    # 閾値以上のすべてのデータ点のインデックスを取得
    indices = np.where(y_transmittance >= threshold)[0]
    
    if len(indices) == 0:
        return None # 帯域が見つからない場合はNoneを返す
        
    return indices


def calculate_bandwidth_nm(x_wavelength, y_transmittance, db_down):
    """指定されたdBダウンでの帯域幅を物理単位(nm)で計算する。"""
    indices = _find_band_indices(y_transmittance, db_down)
    
    if indices is None:
        return 0.0

    # 帯域の左端と右端の波長から幅を計算
    left_boundary_wavelength = x_wavelength[indices[0]]
    right_boundary_wavelength = x_wavelength[indices[-1]]
    
    return right_boundary_wavelength - left_boundary_wavelength



def evaluate_insertion_loss(graph): #グラフ内の最大値を返す　挿入損失計算用
    return np.amax(graph)           

#もともとのやつ
"""
def evaluate_ripple(graph): #3dB波長帯域での極大と極小の差を返す リプル計算用(  グラフの真ん中がトップである前提)
    start_number = int(len(graph)/2)                        #真ん中にトップが来ると仮定している
    length = evaluate_3dbband(graph)                        #3db波長帯域の長さを得る
    point = _local_maximum_and_minimum(graph[start_number - int(length/2)-1:start_number + int(length/2)])  #グラフ内の極大極小をとるx座標を得る
    if not point[0][0] or not point[0][1]:                  #もしなければリプル0とする
        return 0
    ripple = max(point[1][0]) - min(point[1][1])            #リプル＝最も大きい極大値-最も小さい極小値
    return ripple
"""


def evaluate_ripple_improved(graph_db):
    """3dB帯域幅内のリップルを計算する改善版。"""
    # 3dB帯域内のインデックスを取得
    indices_3db = _find_band_indices(graph_db, 3.0)

    if indices_3db is None or len(indices_3db) < 2:
        return 0.0 # 帯域が定義できなければリップル0

    # 3dB帯域内のデータスライスを取得
    pass_band_slice = graph_db[indices_3db[0]:indices_3db[-1]+1]
    
    # スライス内の最大値と最小値の差をリップルとする
    ripple = np.max(pass_band_slice) - np.min(pass_band_slice)
    return ripple




def evaluate_crosstalk(graph):  #3db波長帯域外かつ右半分での傾きが正なっているところのy座標の最大値を返す(グラフの真ん中がトップである前提)
    start_number = int(len(graph)/2)                        #真ん中にトップが来ると仮定している
    length = evaluate_3dbband(graph)                        #3db波長帯域の長さを得る
    temp1 = 0                                               #仮格納先
    temp2 = 0
    for i in range(len(graph)):
        temp1 = temp2
        temp2 = graph[start_number + int(length/2) + i]     #真ん中のx座標＋3dB波長帯域の長さからスタート
        if temp2 > temp1:                                   ##初めての極大を検出
            point = start_number + int(length/2) + i        
            break
    return np.amax(graph) - max(graph[point:])              #クロストークの計算をして返す

def evaluate_shape_factor(graph):                           #国分先生の定義通りの計算
    return _onedb(graph)/_tendb(graph)


def evaluate_crosstalk_improved(graph_db):
    """3dB帯域幅外のクロストークを計算する改善版。"""
    # 3dB帯域のインデックスを取得して、阻止域を定義
    indices_3db = _find_band_indices(graph_db, 3.0)

    if indices_3db is None:
        # パスバンドがない場合、クロストークは評価不能
        return 0.0 

    # 阻止域のスライスを作成
    stop_band_left = graph_db[:indices_3db[0]]
    stop_band_right = graph_db[indices_3db[-1]+1:]
    
    # 左右の阻止域で最も高いサイドローブ（ピーク）の値を見つける
    highest_sidelobe = -np.inf # マイナス無限大で初期化
    if len(stop_band_left) > 0:
        highest_sidelobe = max(highest_sidelobe, np.max(stop_band_left))
    if len(stop_band_right) > 0:
        highest_sidelobe = max(highest_sidelobe, np.max(stop_band_right))
        
    if highest_sidelobe == -np.inf:
        return 999.0 # 阻止域がない場合（理想的すぎる）

    # クロストーク = 主ピークと最も高いサイドローブの差
    crosstalk = np.max(graph_db) - highest_sidelobe
    return crosstalk







def evaluate_3dbband(graph):        #最大値から-3したところの交点の長さを出力
    y = evaluate_insertion_loss(graph)                      #最大値を格納
    start_number = int(len(graph)/2)                        #スタート位置
    for i in range(start_number):                           #最大値から-3となるx座標を得る
        if graph[start_number+i] - y + 3 <0:
            break
    return i*2  #対称性から中心から交点の距離*2としている


def _local_maximum_and_minimum(graph):   #グラフ内の極大極小をとるx座標を得る　リプル計算用
    temp1 = 0                           
    temp2 = -100                        #y座標の閾値
    max_point_x = []
    min_point_x = []
    max_point_y = []
    min_point_y = []
    clock = 0
    for i in range(len(graph)):
        temp1 = temp2
        temp2 = graph[i]
        if temp2 < temp1 and clock == 0:
            max_point_x.append(i)
            max_point_y.append(graph[i])
            clock = 1
        elif temp2 > temp1 and clock == 1:
            min_point_x.append(i)
            min_point_y.append(graph[i])
            clock = 0
    point_x = []
    point_x.append(max_point_x)
    point_x.append(min_point_x)
    point_y = []
    point_y.append(max_point_y)
    point_y.append(min_point_y)
    point = []
    point.append(point_x)
    point.append(point_y)

    return point    #返り値は2*2行列

def _onedb(graph):  #shapefactor用　3db波長帯域計算と同じ原理
    y = evaluate_insertion_loss(graph)
    temp = int(evaluate_3dbband(graph)/2)
    start_number = int(len(graph)/2)
    for i in range(temp):
        if graph[start_number-temp+i] - y + 1 <0:
            break
    return (temp - i)*2

def _tendb(graph):  #shapefactor用　3db波長帯域計算と同じ原理
    y = evaluate_insertion_loss(graph)
    temp = int(evaluate_3dbband(graph)/2)
    start_number = int(len(graph)/2)
    for i in range(temp):
        if graph[start_number-temp-i] - y + 10 <0:
            break
    return (temp +i)*2


# ---------------------------------------------------------------- #
# ▼▼▼ メインの実行部分 ▼▼▼
# ---------------------------------------------------------------- #

# 1. シミュレーションの実行
x_m = calculate_x(center_wavelength=1550e-9, FSR=35e-9)
graph_db = simulate_transfer_function(
    wavelength=x_m,
    L=np.array([
        6.2e-05, 6.2e-05, 7.75e-05, 7.75e-05, 7.75e-05, 7.75e-05,
    ]),
    K=np.array([0.1019, 0.0255, 0.0240, 0.0397, 0.0859, 0.3755, 0.9292]),
    alpha=11.51,
    eta=0.996,
    n_eff=2.2,
    n_g=4.4,
    center_wavelength=1550e-9
)

# 2. 改善された評価関数でグラフの特性を評価
# x軸をメートル(m)からナノメートル(nm)に変換して渡す
x_nm = x_m * 1e9
evaluate_graph_improved(x_nm, graph_db)


