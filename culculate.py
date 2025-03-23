import numpy as np
from MRR.simulator import (
    calculate_practical_FSR,
    calculate_ring_length,
    calculate_x,
    optimize_N,
)
from MRR.transfer_function import simulate_transfer_function




def evaluate_graph(graph):                                  #まとめて出力するためのもの
    evaluate_data = [evaluate_insertion_loss(graph),evaluate_3dbband(graph)*0.01,evaluate_ripple(graph),evaluate_crosstalk(graph),evaluate_shape_factor(graph)]
    print("insertion loss = {} dB".format(evaluate_data[0]))
    print("3dbband = {} nm".format(evaluate_data[1]))
    print("ripple = {} dB".format(evaluate_data[2]))
    print("crosstalk = {} dB".format(evaluate_data[3]))
    print("shape_factor = {}".format(evaluate_data[4]))
    return

def evaluate_insertion_loss(graph): #グラフ内の最大値を返す　挿入損失計算用
    return np.amax(graph)           

def evaluate_ripple(graph): #3dB波長帯域での極大と極小の差を返す リプル計算用(  グラフの真ん中がトップである前提)
    start_number = int(len(graph)/2)                        #真ん中にトップが来ると仮定している
    length = evaluate_3dbband(graph)                        #3db波長帯域の長さを得る
    point = _local_maximum_and_minimum(graph[start_number - int(length/2)-1:start_number + int(length/2)])  #グラフ内の極大極小をとるx座標を得る
    if not point[0][0] or not point[0][1]:                  #もしなければリプル0とする
        return 0
    ripple = max(point[1][0]) - min(point[1][1])            #リプル＝最も大きい極大値-最も小さい極小値
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

x = calculate_x(center_wavelength=1550e-9, FSR=35e-9)
graph = simulate_transfer_function(
    wavelength = x,
    L = np.array(
        [
        7.749999999999999e-05,
        6.2e-05,
        6.2e-05,
        7.749999999999999e-05,
        7.749999999999999e-05,
        7.749999999999999e-05,
        ]
    ),
    K = np.array([0.54331948, 0.07410394, 0.02948469, 0.01517046, 0.11400472, 0.06921048, 0.43457259]),
    alpha = 52.96,
    eta = 0.996,
    n_eff=2.2,
    n_g=4.4,
    center_wavelength = 1550e-9
)


evaluate_graph(graph)
