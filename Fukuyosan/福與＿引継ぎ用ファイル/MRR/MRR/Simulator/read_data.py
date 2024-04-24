import csv

def read_K(n:int,number_of_rings:int):                                              #Kの読み込み
    file_name = "MRR/data/K" + str(number_of_rings) + "_" + str(n) +".csv"          #ファイル名の読み込み
    with open(file_name) as file:                                                   #ファイル展開
        reader = csv.reader(file,quoting =csv.QUOTE_NONNUMERIC)
        data = [row for row in reader]
    return data                                                                     #行ごとのデータを配列で返す

def read_L(n:int,number_of_rings:int):                                              #Lの読み込み
    file_name = "MRR/data/L" + str(number_of_rings) + "_" + str(n) +".csv"          #ファイルの読み込み
    with open(file_name) as file:                                                   #ファイル展開
        reader = csv.reader(file,quoting =csv.QUOTE_NONNUMERIC)
        data = [row for row in reader]
    return data     