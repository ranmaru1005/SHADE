import matplotlib.pyplot as plt
import numpy as np

def graph_cretate(axis,dataset,nameset): #datasetは配列　axisはx軸　namesetは配列で凡例用の名前　Lとsは実験用なので削除予定　一枚表示or二枚重ねる用の関数です
    if len(dataset) == 1:
        _onegraph_create(axis,dataset[0],nameset[0])
        _onetopgraph_create(axis,dataset[0],nameset[0])
    elif len(dataset) == 2:
        _twograph_create(axis,dataset[0],dataset[1],nameset[0],nameset[1])
        _twotopgraph_create(axis,dataset[0],dataset[1],nameset[0],nameset[1])


def _onegraph_create(axis,data1,name1):     #一つのグラフのみ表示
    plt.rcParams["xtick.direction"]="in"    #目盛り内向き
    plt.rcParams["ytick.direction"]="in"    #目盛り内向き
    plt.plot(axis,data1,label=name1)        #プロット
    plt.xlabel("Wavelength (nm)",fontsize=13)       #x軸ラベル
    plt.ylabel("Transmittance (dB)",fontsize = 13)  #y軸ラベル
    plt.ylim(-40,0)                                 #y軸表示範囲-60~0dB
    plt.xticks(np.arange(1540,1565,5))              #x軸目盛りの表示位置を設定
    plt.minorticks_on()                             #補助目盛オン
    plt.legend(bbox_to_anchor=(1,1),loc="upper right")  #凡例を右上に表示　locだけでなく位置を変えたいならanchorも変える必要ありなため、調べてください
    plt.show()                                          #グラフ表示

def _onetopgraph_create(axis,data1,name1):  #一つのグラフをトップの部分を拡大して表示
    plt.rcParams["xtick.direction"]="in"    #目盛り内向き
    plt.rcParams["ytick.direction"]="in"    #目盛り内向き
    plt.plot(axis,data1,label=name1)        #プロット
    plt.xlabel("Wavelength (nm)",fontsize = 13)     #x軸ラベル
    plt.ylabel("Transmittance (dB)",fontsize = 13)  #y軸ラベル
    plt.ylim(-6,0)                                 #y軸表示範囲-6~0dB
    plt.xlim(1549,1551)                            #x軸表示範囲1549nm~1551nm
    plt.xticks(np.arange(1549,1551.25,0.5))         #x軸目盛りの表示位置を設定
    plt.minorticks_on()                             #補助目盛オン
    plt.legend(bbox_to_anchor=(1,1),loc="upper right")  #凡例を右上に表示　locだけでなく位置を変えたいならanchorも変える必要ありなため、調べてください
    plt.show()                                          #グラフ表示

def _twograph_create(axis,data1,data2,name1,name2):     #二つのグラフを重ねて表示
    plt.rcParams["xtick.direction"]="in"    #目盛り内向き
    plt.rcParams["ytick.direction"]="in"    #目盛り内向き
    plt.plot(axis,data1,label=name1)        #一つ目のグラフをプロット
    plt.plot(axis,data2,label=name2,linestyle = "dashed")        #二つ目のグラフをプロット
    plt.xlabel("Wavelength (nm)",fontsize = 13)     #x軸ラベル
    plt.ylabel("Transmittance (dB)",fontsize = 13)  #y軸ラベル
    plt.ylim(-40,0)                                 #y軸表示範囲-60~0dB
    plt.xticks(np.arange(1540,1565,5))              #x軸目盛りの表示位置を設定
    plt.minorticks_on()                             #補助目盛オン
    plt.legend(bbox_to_anchor=(1,1),loc="upper right")  #凡例を右上に表示　locだけでなく位置を変えたいならanchorも変える必要ありなため、調べてください
    plt.show()              #グラフ表示

def _twotopgraph_create(axis,data1,data2,name1,name2):
    plt.rcParams["xtick.direction"]="in"    #目盛り内向き
    plt.rcParams["ytick.direction"]="in"    #目盛り内向き
    plt.plot(axis,data1,label=name1)       #一つ目のグラフをプロット
    plt.plot(axis,data2,label=name2,linestyle = "dashed")       #二つ目のグラフをプロット
    plt.xlabel("Wavelength (nm)",fontsize = 13)     #x軸ラベル
    plt.ylabel("Transmittance (dB)",fontsize = 13)     #y軸ラベル
    plt.ylim(-6,0)                                 #y軸表示範囲-6~0dB
    plt.xlim(1549,1551)                            #x軸表示範囲1549nm~1551nm
    plt.xticks(np.arange(1549,1551.25,0.5))         #x軸目盛りの表示位置を設定
    plt.minorticks_on()                             #補助目盛オン
    plt.legend(bbox_to_anchor=(1,1),loc="upper right")  #凡例を右上に表示　locだけでなく位置を変えたいならanchorも変える必要ありなため、調べてください
    plt.show()              #グラフ表示

