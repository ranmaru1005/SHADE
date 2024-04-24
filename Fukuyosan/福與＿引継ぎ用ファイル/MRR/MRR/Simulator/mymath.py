import numpy as np
import math
import tensorflow as tf
import random

def minus(x):                                       #map関数とあわせて用いる　負をかけて小数第5位までを切り捨て
    return math.floor(-x * 10 ** 4) / (10 ** 4)

def minus_and_round(x):                             #負をかけて、小数第5位までを切り捨て
    return round(math.floor(-x * 10 ** 4) / (10 ** 4),0)

def minus_and_round_and_random(x):                             #負をかけて、小数第5位までを切り捨て
    return math.floor(x * 10 ** 4) / (10 ** 4) + random.uniform(-0.5,0.5)

def mean_percentage_squared_error(y_true,y_pred):   #平均二乗パーセント誤差　未使用
    percentage_error = (y_true - y_pred) / y_true
    loss = tf.math.reduce_mean(percentage_error**2)
    return loss


def graph_integrate(data1:np.array,data2:np.array,boader): #積分値の計算　boaderの設定値によって一定以上の部分のみ積分計算をする
    if len(data1)==len(data2):
        area = 0
        for i in range(len(data1)):
                if data1[i]>data2[i]:
                    if data2[i]>-boader:
                        area += abs((1/data1[i])*(data1[i]-data2[i]))
                    elif data1[i]>-boader:
                        area += abs((1/data1[i])*(data1[i]+boader))
                    else:
                        pass
                if data2[i]>data1[i]:
                    if data1[i]>-boader:
                        area += abs((1/data2[i])*(data2[i]-data1[i]))
                    elif data2[i]>-boader:
                        area += abs((1/data2[i])*(data2[i]+boader))
                    else:
                        pass
        return area
    else:
        print("length of datas is not match")               #互いの配列の長さが一致していない場合は何もせず返す
        return