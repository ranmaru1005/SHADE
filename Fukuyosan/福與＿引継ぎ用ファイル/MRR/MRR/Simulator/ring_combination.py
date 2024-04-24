import csv
import numpy as np
import itertools

def _read_csv_values(file_name):
    with open(file_name) as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        data = [row for row in reader]
    return data               

def _generate_fixed_length_combinations(a, b, number_of_rings):
    combinations = itertools.product([a, b], repeat=number_of_rings)
    # 全て同じ要素で構成される配列を除外
    filtered_combinations = [comb for comb in combinations if not all(x == comb[0] for x in comb)]
    return np.array(filtered_combinations)

def generate_allcomb_Lcsv(number_of_rings):
    # CSVファイルから値を読み込む
    values = _read_csv_values("MRR/data/FSR20nm_ring_combinations.csv")

    # すべての組み合わせを格納するリスト
    all_combinations = []

    # 組み合わせを生成
    for i in range(len(values)):
        combinations_array = _generate_fixed_length_combinations(values[i][0], values[i][1], number_of_rings)
        for combination in combinations_array:
            all_combinations.append(combination)

    # すべての組み合わせを一つのCSVファイルに書き込む
    file_name = "MRR/data/combination_test2/allcomb_FSR20nm_L" + str(number_of_rings) + ".csv"
    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file)
        for combination in all_combinations:
            writer.writerow(combination)

