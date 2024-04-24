from tensorflow.python.keras.models import load_model

def load_NN(n:int,number_of_rings:int):                                                     #ニューラルネットワークの読み込み
    file_name = "MRR/data/DNN2_K_" + str(number_of_rings) + "_" + str(n) +".h5"     #ファイル名
    model_DNN = load_model(file_name)
    return model_DNN   