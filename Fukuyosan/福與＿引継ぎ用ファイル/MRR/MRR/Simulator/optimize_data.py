import numpy as np
import csv
from mymath import minus_and_round,graph_integrate
from transfer_function import TransferFunction
from load_NN import load_NN
from read_data import read_K
import copy

def pre_K_NN(n:int,number_of_rings:int,L:np.array):                                 #Kを予測するためのニューラルネットワーク
    pre_number = 10                                                                 #予測値を10通り返す
    xaxis = np.arange(1540e-9,1560e-9,0.01e-9)                                      #シミュレーション範囲1540nm~1560nmを0.01nm間隔でプロットするためのx軸用の配列
    model_DNN = load_NN(n,number_of_rings)                                          #モデルのロード
    pre_data_K = np.array(read_K(pre_number,number_of_rings))                       #予測に用いる入力K
    file_name = "MRR/data/pred_K" + str(number_of_rings) + "_" + str(n) +".csv"     #結果の保存先のファイル名
    with open(file_name,"w",newline="") as file:                                    #ファイル展開
        writer = csv.writer(file)                                                   #書き込み用
        for i in range(pre_number):
            pre_data = TransferFunction(L,pre_data_K[i],config={'center_wavelength':1550e-9,'eta':0.996,'n_eff':2.2,'n_g':4.4,'alpha':52.96})   #シミュレーションデータを格納
            temp = np.array(list(map(minus_and_round,pre_data.simulate(xaxis))))    #NNに入力するために加工
            pred_Y = model_DNN.predict(temp.reshape(1,len(xaxis)))                  #予測値を格納
            print(pred_Y[0],pre_data_K[i])                                          #結果を表示
            writer.writerow(np.append(pred_Y[0],pre_data_K[i]))                     #結果のデータを書き込み

def search_K(L,K,xaxis,target_trans_data,boader):                                          #targetには目標とする伝達関数の配列　LとKは目標に近づけたい補正前の配列　補正法は勾配法を用いている
    ans_K = 0                                                                       
    true_K = copy.copy(K)                                                           
    temp = 10                                                                       #ans_K及びtrue_Kは仮格納先　tempは探索繰り返し回数
    data = TransferFunction(L,K,config={'center_wavelength':1550e-9,'eta':0.996,'n_eff':2.2,'n_g':4.4,'alpha':52.96})       #シミュレーション
    trans_data = data.simulate(xaxis)                                               #伝達関数仮格納先
    
    for i in range(len(true_K)):                                                    #for内は勾配法の計算
        temp_K = copy.copy(true_K)
        for j in range(temp*2+1):
            if 0.01<true_K[i]<0.99:                                                 #0.01<K<0.99内だったら格納　そうでなければ計算続行
                temp_K[i] = true_K[i] + 0.01*(temp-j)
            if temp_K[i]<0.01 or  temp_K[i]>0.99:
                continue
            data = TransferFunction(L,temp_K,config={'center_wavelength':1550e-9,'eta':0.996,'n_eff':2.2,'n_g':4.4,'alpha':52.96})
            trans_data = data.simulate(xaxis)
            temp_S = graph_integrate(target_trans_data,trans_data,boader)       #targetと現在のLとtemp_Kによる伝達関数で囲われた部分の面積を出す
            if j == 0:                                                              #K[i]についての探索終了時の面積を格納
                S=temp_S
            elif S > temp_S:                                                        #より面積が小さくなるK[i]ならば、そのK[i]及び面積を格納
                S=temp_S
                ans_K = temp_K[i]
        if ans_K != 0:                                                              #K[i]の値を確定、なければそのまま
            true_K[i] = ans_K
    data = TransferFunction(L,true_K,config={'center_wavelength':1550e-9,'eta':0.996,'n_eff':2.2,'n_g':4.4,'alpha':52.96})
    trans_data = data.simulate(xaxis)
    temp_S = graph_integrate(target_trans_data,trans_data,boader)
    return true_K                                                                   #補正後のKを返す

def fix_K(K,number):
    for i in range(len(K)):
        if K[i]>1 or K[i]<0:
            K[i]=number
    return K
