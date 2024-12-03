import numpy as np
import concurrent.futures
from pyDOE2 import lhs
import random

def shade(objective_function, number_of_rings, eta=0.996, pop_size=20, gen=500, tol=1e-6, seed=None, workers=4, params=None):
    np.random.seed(seed)
    random.seed(seed)
    
    dim = number_of_rings + 1
    min_val = 1e-12
    max_val = eta
    
    # 1. 初期集団生成（ラテンハイパーキューブサンプリング）
    lhs_samples = lhs(dim, samples=pop_size, criterion='maximin')
    population = min_val + (max_val - min_val) * lhs_samples
    fitness_values = [objective_function(ind, params) for ind in population]
    
    # 初期化
    memory_size = 25  # 成功履歴の保存数
    memory_cr = np.full(memory_size, 0.5)  # 交叉率
    memory_f = np.full(memory_size, 0.5)   # スケールファクター
    archive = []  # 選択から除外された個体を保持
    
    best_idx = np.argmin(fitness_values)
    best_individual = population[best_idx]
    best_fitness = fitness_values[best_idx]
    
    for g in range(gen):
        new_population = []
        new_fitness_values = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i, target in enumerate(population):
                # 適応的なFとCRの生成
                idx = random.randint(0, memory_size - 1)
                cr = np.clip(np.random.normal(memory_cr[idx], 0.1), 0, 1)
                f = 0
                while f <= 0 or f > 1:
                    f = np.random.normal(memory_f[idx], 0.1)
                
                # 変異操作: current-to-pbest/1
                p = int(np.ceil(pop_size * 0.2))  # 上位20%の個体
                pbest_idx = np.random.choice(np.argsort(fitness_values)[:p])
                pbest = population[pbest_idx]
                
                candidates = [ind for j, ind in enumerate(population) if j != i]
                if archive:
                    candidates += archive
                a, b = random.sample(candidates, 2)
                
                mutant = target + f * (pbest - target) + f * (a - b)
                mutant = np.clip(mutant, min_val, max_val)
                
                # 交叉操作
                trial = np.where(np.random.rand(dim) < cr, mutant, target)
                trial = np.clip(trial, min_val, max_val)
                
                # 並列で子個体を評価
                futures.append(executor.submit(objective_function, trial, params))
            
            # 子個体の評価結果を収集
            for i, future in enumerate(futures):
                trial_fitness = future.result()
                if trial_fitness < fitness_values[i]:
                    new_population.append(trial)
                    new_fitness_values.append(trial_fitness)
                    
                    # 成功履歴を更新
                    memory_cr[idx] = 0.9 * memory_cr[idx] + 0.1 * cr
                    memory_f[idx] = 0.9 * memory_f[idx] + 0.1 * f
                    
                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness
                else:
                    new_population.append(population[i])
                    new_fitness_values.append(fitness_values[i])
        
        # 更新と収束判定
        population = new_population
        fitness_values = new_fitness_values
        
        fitness_std = np.std(fitness_values)
        if fitness_std < tol:
            print(f"Converged at Generation {g}: Best Fitness = {best_fitness}, Std = {fitness_std}")
            break
        print(f"Generation {g}: Best Fitness = {best_fitness}, Std = {fitness_std}")
    
    return best_individual, best_fitness
