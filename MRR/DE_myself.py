import random
import numpy as np
from pyDOE2 import lhs  # LHSサンプリング用
from multiprocessing import Pool
from functools import partial

def differential_evolution(
    objective_function, number_of_rings, eta=0.996, pop_size=20, gen=500, CR=0.5, F=0.5, tol=1e-6, seed=None, workers=4, params=None
):
    print("params =",params)
    """
    Parallel Differential Evolution (DE) with LHS and convergence criteria, supporting extra parameters.

    Parameters:
    - objective_function: 最適化したい目的関数 (x, params)
    - number_of_rings: 各個体の次元数 - 1 を加えた値
    - eta: 各個体の最大値
    - pop_size: 集団サイズ
    - gen: 最大世代数
    - CR: 交叉率
    - F: スケールファクター
    - tol: 許容誤差 (収束判定用)
    - seed: 乱数シード
    - workers: 並列処理に使用するプロセス数
    - params: objective_function に渡す追加パラメータ (OptimizeKParams)
    
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
    
    # 固定引数を持つ関数を生成
    evaluate = partial(objective_function, params=params)
    
    # 初期集団の適応度を並列で計算
    with Pool(processes=workers) as pool:
        fitness_values = pool.map(evaluate, population)

    # 最良の個体を初期設定
    best_idx = np.argmin(fitness_values)
    best_individual = population[best_idx]
    best_fitness = fitness_values[best_idx]

    # 2. DEのメインループ
    for g in range(gen):
        new_population = []
        trials = []
        
        for i, target in enumerate(population):
            # 変異操作: current-to-best/1
            candidates = [ind for j, ind in enumerate(population) if j != i]
            a, b = random.sample(candidates, 2)
            mutant = target + F * (best_individual - target) + F * (a - b)
            mutant = np.clip(mutant, min_val, max_val)
            
            # 交叉操作
            trial = np.where(np.random.rand(dim) < CR, mutant, target)
            trial = np.clip(trial, min_val, max_val)
            trials.append(trial)
        
        # 子個体の適応度を並列で計算
        with Pool(processes=workers) as pool:
            trial_fitness_values = pool.map(evaluate, trials)

        for i, trial_fitness in enumerate(trial_fitness_values):
            if trial_fitness < fitness_values[i]:
                new_population.append(trials[i])
                fitness_values[i] = trial_fitness
                if trial_fitness < best_fitness:
                    best_individual = trials[i]
                    best_fitness = trial_fitness
            else:
                new_population.append(population[i])

        # 集団の更新
        population = new_population
        
        # 収束条件の確認
        fitness_std = np.std(fitness_values)
        if fitness_std < tol:
            print(f"Converged at Generation {g}: Best Fitness = {best_fitness}, Std = {fitness_std}")
            break
        print(f"Generation {g}: Best Fitness = {best_fitness}, Std = {fitness_std}")

    return best_individual, best_fitness
