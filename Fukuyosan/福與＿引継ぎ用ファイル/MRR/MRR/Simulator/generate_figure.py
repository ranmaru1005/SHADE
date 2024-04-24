import matplotlib.pyplot as plt
import numpy as np


def base_array(n,bottom):   #仮で用意する配列
    return [bottom]*n

def set_3db(center,db_length,top,data): #トップの部分を形成
    for i in range(db_length):
        data[int(center-int(db_length/2)+i)] = top

def set_cross(center,length,cross,data):   #サイドローブのトップを形成
    if cross == 0:
        return
    for i in range(length):
        data[int(center-int(length/2)+i+656)] = cross
        data[int(center-int(length/2)+i-655)] = cross

def set_cross2(length,cross2,data):     #2つ目のサイドローブのトップを形成
    if cross2 == 0:
        return
    for i in range(int(length/2)):
        data[i] = cross2
        data[-i] = cross2

def set_slope_fromcrosstocenter(center,db_length,cross_length,cross,data):#トップとサイドローブの間のスロープ形成
    bottomtop = 105 #100 #300 #67.5  #トップとサイドローブの間の頂点のy座標
    half_db_length = int(db_length/2)
    half_cross_length = int(cross_length/2)
    top = int((center + half_db_length + center-half_cross_length + 656)/2)
    a1 = (-3+bottomtop)/((center + half_db_length-top)**2)
    a2 = (cross+bottomtop)/((center - half_cross_length + 656 - top)**2)
    for i in range (top - (center + half_db_length)):
        data [center + half_db_length + i] = a1*((center + half_db_length + i - top)**2) - bottomtop
        data [center - half_db_length - i] = data [center + half_db_length + i]
    for j in range (center - half_cross_length + 656 - top):
        data [top + j] = a2*((j)**2) - bottomtop
        data [2*center - top - j] = data [top + j]

def set_slope_fromcrosstocross(center,cross_length,cross,cross2,data):#サイドローブ間のスロープ部分の形成
    bottomtop = 62.5 #100 #50 #105 #55 #62.5
    half_cross_length = int(cross_length/2)
    top = int((center + half_cross_length + 656 + 2001 - cross_length)/2)
    a1 = (cross+bottomtop)/((center + half_cross_length + 656 - top)**2)
    a2 = (cross2+bottomtop)/((2001 - half_cross_length - top)**2)
    for i in range (top - (center + half_cross_length + 611)):
        data [center + half_cross_length + 656 + i] = a1*((center + half_cross_length + 656 + i - top)**2) - bottomtop
        data [center - half_cross_length - 656 - i] = data [center + half_cross_length + 656 + i]
    for j in range (2001 - half_cross_length - top + 1):
        data [top + j] = a2*((j)**2) - bottomtop
        data [2*center - top - j] = data [top + j]

def set_ripple(center,position,ripple): #未実装、理想的なフィルタにおいて必要性は無い
    return center

def generate_figure(center,db_length,cross_length,cross,cross2,number_of_array,bottom,top):
    data = base_array(number_of_array,bottom)
    set_3db(center,db_length,top,data)
    set_cross(center,cross_length,cross,data)
    set_cross2(cross_length,cross2,data)
    set_slope_fromcrosstocenter(center,db_length,cross_length,cross,data)
    set_slope_fromcrosstocross(center,cross_length,cross,cross2,data)

    return data  

# 使い方

# axis = np.arange(1540,1560.01,0.01) 
# plt.plot(axis,generate_figure(center = 1000,db_length = 80,cross_length = 80,cross = -44,cross2 = -39,number_of_array = 2001,bottom = -100,top = -3))
# plt.show()