import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import random
import os
from mymath import minus
from transfer_function import TransferFunction


#以下研究で用いていた初期設定
# L=np.array([82.4e-6,82.4e-6,55.0e-6,55.0e-6])
# number_of_rings = 4                                 #リング数
# n = 100000                                          #データ数
# data_K = np.array(read_K(n,number_of_rings))        #Kの格納先　DNNの教師データ

# xaxis = np.arange(1540e-9,1560e-9,0.01e-9)          #シミュレーション範囲1.54µm~1.56µm
# epochs = 200                                        #訓練回数
# batch_size = 1000                                   #学習時に一度に計算するデータ数

def DNN_K(L,number_of_rings,n,data_K,xaxis,epochs,batch_size):  #Kを予測するDNNを作成する関数
    input_data = []                                     #DNNに用いるデータの仮格納先
    #データセットの用意
    for i in range(n):  #伝達関数と結合率Ｋの対応関係を学習させるため、XとYでそれぞれ格納
        trans_data = TransferFunction(L,data_K[i],config={'center_wavelength':1550e-9,'eta':0.996,'n_eff':2.2,'n_g':4.4,'alpha':52.96})
        input_data.append(list(map(minus,trans_data.simulate(xaxis))))  #データを学習向けに加工している。伝達関数は基本負であるが、正の値の方が学習がうまくいくためmap関数で配列全体を正にしている
        train_X = np.reshape(input_data,(n,len(xaxis)))                 #DNNに入力できるように配列の形を変えている
        train_Y = np.reshape(data_K,(n,number_of_rings+1))              #DNNに入力できるように配列の形を変えている

    #シード値の固定　再現性があることを示す
    seed = 1
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    #モデルの構造　入力層1 中間層3 出力層1　活性化関数は回帰を期待して線形(identityとも呼ぶ)
    model_DNN = Sequential()                                                            #Sequentialという鎖型のモデルを作成
    model_DNN.add(Dense(len(xaxis), input_shape = (len(xaxis),), activation='linear'))  #層はDenseという種類で、dense(入力数、形、活性化関数) input_shapeの部分は試行錯誤の末によるもの
    model_DNN.add(Dropout(0.2))                                                         #過学習を抑制するために学習を20%の確率で忘れさせる
    model_DNN.add(Dense(1250,activation='linear'))
    model_DNN.add(Dropout(0.2))
    model_DNN.add(Dense(1250,activation='linear'))
    model_DNN.add(Dropout(0.1))
    model_DNN.add(Dense(1250,activation='linear'))
    model_DNN.add(Dropout(0.1))
    model_DNN.add(Dense(number_of_rings+1, activation='linear'))
    model_DNN.compile(optimizer = Adam(lr=0.001), loss="mean_squared_error", metrics="accuracy")    #オプティマイザーはAdamで学習率は0.1%　損失関数は平均二乗誤差

    #モデルの構造を表示する
    print(model_DNN.summary())

    #CallBacks  最も学習がうまくいったところを呼び戻して保存する
    filename = "MRR/data/DNN2_K_" + str(number_of_rings) + "_" + str(n) +".h5"
    checkpoint = ModelCheckpoint(filepath=filename, monitor="loss",verbose=0,save_best_only=True,save_weights_only=False,mode="min",period=1)   #lossを監視し、mode minよりlossが一番低いところで保存

    #学習開始
    history = model_DNN.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.01,callbacks=[checkpoint])

    #学習度のグラフ表示及び保存
    plt.plot(range(epochs),history.history["loss"],label="loss")                                #lossをプロット
    plt.plot(range(epochs),history.history["val_loss"],label="val_loss")                        #val_lossをプロット
    plt.xlabel("epoch")                                                                         #x軸ラベル
    plt.ylabel("loss")                                                                          #y軸ラベル
    plt.ylim(0,20)                                                                              #y軸の表示範囲
    plt.legend(bbox_to_anchor=(1,0),loc="lower right")                                          #右下に凡例
    plt.savefig("MRR/data/DNN_K_" + str(number_of_rings) + "_" + str(n) + "_" + "figure.jpg")   #グラフ保存
    plt.show()                                                                                  #グラフ表示