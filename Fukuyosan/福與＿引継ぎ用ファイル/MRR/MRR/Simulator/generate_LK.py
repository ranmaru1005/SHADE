import numpy as np
import math
import csv

def _floor_number(list,n):   #丸め込み関数
    for i in range (len(list)):
        list[i] = math.floor(list[i] * 10 ** n) / (10 ** n)
    return list

def generate_L(n,number_of_rings):                                              #Lを生成してCSVファイルとして保存
    file_name = "MRR/data/L" + str(number_of_rings) + "_" + str(n) +".csv"
    with open(file_name,"w",newline="") as file:
        writer = csv.writer(file)
        for i in range(n):
            L = _floor_number(np.random.uniform(50e-6,150e-6,number_of_rings),7) #50µ～150µの範囲で小数7位までの乱数をリングの数だけnparrayとして生成
            writer.writerow(L)

def generate_K(n,number_of_rings):                                              #Kを生成してCSVファイルとして保存
    file_name = "MRR/data/K" + str(number_of_rings) + "_" + str(n) +".csv"
    with open(file_name,"w",newline="") as file:
        writer = csv.writer(file)
        for i in range(n):
            K = _floor_number(np.random.uniform(0.05,0.60,number_of_rings+1),2) #0.05～0.6までの小数2位までの乱数をリングの数だけnparrayとして生成
            writer.writerow(K)