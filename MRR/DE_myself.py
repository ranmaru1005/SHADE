import random
import numpy as np

def differential_evolution(objective_function, params, number_of_rings,  eta=0.996, pop_size=15, gen=500, CR=0.5, F=0.5):
    print("パラメータ確認",params)
    """
    Differential Evolution (DE) アルゴリズムの実装。
    
    Parameters:
    - objective_function: 最適化したい目的関数（評価関数）
    - number_of_rings: 集団内の各個体の次元数を決定する値
    - eta: 各個体の最大値
    - pop_size: 集団サイズ
    - gen: 最大世代数
    - CR: 交叉率
    - F: スケールファクター
    
    Returns:
    - best_individual: 最適な個体
    - best_fitness: 最適な個体の適応度
    """
    
    # 次元数の設定
    dim = number_of_rings + 1
    min_val = 1e-12  # 各要素の最小値
    max_val = 0.996    # 各要素の最大値
    
    # 1. 初期集団の生成
    population = [np.random.uniform(min_val, max_val, dim) for _ in range(pop_size)]
    fitness_values = [objective_function(ind, params) for ind in population]
    
    # 最良の個体を初期設定
    best_idx = np.argmin(fitness_values)
    best_individual = population[best_idx]
    best_fitness = fitness_values[best_idx]
    
    # 2. DEのメインループ
    for g in range(gen):
        print("いまの世代 = ",g)
        new_population = []
        
        for i, target in enumerate(population):
            # 変異操作: current-to-best/1
            # 3つのランダムな個体を選択（ターゲット個体は除く）
            candidates = [ind for j, ind in enumerate(population) if j != i]
            a, b = random.sample(candidates, 2)
            
            # 突然変異ベクトルの生成
            mutant = target + F * (best_individual - target) + F * (a - b)
            
            # クリッピング処理で範囲制約を適用
            mutant = np.clip(mutant, min_val, max_val)
            
            # 交叉操作
            trial = [mutant[d] if random.random() < CR else target[d] for d in range(dim)]
            trial = np.clip(trial, min_val, max_val)
            
            # 子個体の評価
            trial_fitness = objective_function(trial, params)
            
            # 選択操作: 子個体と親個体の適応度を比較
            if trial_fitness < fitness_values[i]:
                new_population.append(trial)
                fitness_values[i] = trial_fitness
                # 最良の個体を更新
                if trial_fitness < best_fitness:
                    best_individual = trial
                    best_fitness = trial_fitness
            else:
                new_population.append(target)

        # 集団の更新
        population = new_population
        for i in population:
            print("解候補 = ",i)
        print(f"Generation {g}: Best Fitness = {best_fitness}")

    # 最適解の出力
    return best_individual, best_fitness
