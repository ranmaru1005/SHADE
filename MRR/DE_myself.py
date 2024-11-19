import random
import numpy as np
from pyDOE2 import lhs  # LHSサンプリング用

def differential_evolution(objective_function, params, number_of_rings, eta=0.996, pop_size=20, gen=500, CR=0.5, F=0.5, seed=None):
    """
    Differential Evolution (DE) with improved initial population generation using LHS.
    
    Parameters:
    - objective_function: 最適化したい目的関数
    - number_of_rings: 各個体の次元数 - 1 を加えた値
    - eta: 各個体の最大値
    - pop_size: 集団サイズ
    - gen: 最大世代数
    - CR: 交叉率
    - F: スケールファクター
    - seed: 乱数シード
    
    Returns:
    - best_individual: 最適な個体
    - best_fitness: 最適な個体の適応度
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # 次元数と探索空間の範囲を設定
    dim = number_of_rings + 1
    min_val = 1e-12
    max_val = eta
    
    # 1. ラテンハイパーキューブサンプリングによる初期集団生成
    lhs_samples = lhs(dim, samples=pop_size, criterion='maximin')  # maximinで均等性を強調
    population = min_val + (max_val - min_val) * lhs_samples  # サンプリング結果を適用範囲にスケーリング
    fitness_values = [objective_function(ind, params) for ind in population]

    # 最良の個体を初期設定
    best_idx = np.argmin(fitness_values)
    best_individual = population[best_idx]
    best_fitness = fitness_values[best_idx]

    # 2. DEのメインループ
    for g in range(gen):
        new_population = []
        for i, target in enumerate(population):
            # 変異操作: current-to-best/1
            candidates = [ind for j, ind in enumerate(population) if j != i]
            a, b = random.sample(candidates, 2)
            mutant = target + F * (best_individual - target) + F * (a - b)
            mutant = np.clip(mutant, min_val, max_val)
            
            # 交叉操作
            trial = np.where(np.random.rand(dim) < CR, mutant, target)
            trial = np.clip(trial, min_val, max_val)
            
            # 子個体の評価
            trial_fitness = objective_function(trial, params)
            
            # 選択操作
            if trial_fitness < fitness_values[i]:
                new_population.append(trial)
                fitness_values[i] = trial_fitness
                if trial_fitness < best_fitness:
                    best_individual = trial
                    best_fitness = trial_fitness
            else:
                new_population.append(target)

        # 集団の更新
        population = new_population
        print(f"Generation {g}: Best Fitness = {best_fitness}")

    return best_individual, best_fitness
