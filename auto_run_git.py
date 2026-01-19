import subprocess
import time

# ==========================================
# 設定エリア: 指定された No.8 ～ No.20 のパターン
# ==========================================
patterns = [
    "ABBAAB",  # 8
    "ABBABA",  # 9
    "ABBBAA",  # 10
    "BBBAAA",  # 11
    "BBABAA",  # 12
    "BBAABA",  # 13
    "BBAAAB",  # 14
    "BABBAA",  # 15
    "BABABA",  # 16
    "BABAAB",  # 17
    "BAABBA",  # 18
    "BAABAB",  # 19
    "BAAABB"   # 20
]

# 各パターンごとの試行回数 (今回は5回)
RUNS_PER_PATTERN = 5

# ==========================================
# 自動実行ロジック
# ==========================================

def run_command(command_list):
    """コマンドを実行し、ログを表示する関数"""
    cmd_str = " ".join(command_list)
    print(f"\n[System] Executing: {cmd_str}")
    try:
        # subprocess.runでコマンドを実行（完了するまで待機）
        result = subprocess.run(command_list, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] Command failed: {cmd_str}")
        print(e)
        return False

def main():
    total_tasks = len(patterns) * RUNS_PER_PATTERN
    current_task = 0

    print(f"==================================================")
    print(f" Start Auto Optimization with Git Push")
    print(f" Patterns: {len(patterns)}")
    print(f" Runs/Pattern: {RUNS_PER_PATTERN}")
    print(f" Total Runs: {total_tasks}")
    print(f"==================================================")

    for pattern in patterns:
        print(f"\n>>>>>>>>>> Starting Pattern: {pattern} <<<<<<<<<<")
        
        for i in range(RUNS_PER_PATTERN):
            current_task += 1
            run_id = i + 1
            print(f"\n--- Progress: {current_task}/{total_tasks} (Pattern: {pattern}, Run: {run_id}/{RUNS_PER_PATTERN}) ---")

            # 1. main.py を実行
            # --skip-plot をつけて高速化します
            success = run_command(["python", "main.py", "--pattern", pattern, "--skip-plot"])
            
            if not success:
                print("Optimization failed. Continuing to next run...")
                continue

            # 2. Git Commit & Push
            
            # resultフォルダを強制的に追加 (-f)
            run_command(["git", "add", "-f", "result"])
            
            # コミットメッセージを作成 (ユーザー指定の形式)
            # 例: 「ABBAABの配置」 (Run 1)
            commit_message = f"「{pattern}の配置」 (Run {run_id})"
            
            # コミット実行
            run_command(["git", "commit", "-m", commit_message])
            
            # プッシュ実行
            push_success = run_command(["git", "push"])
            
            if not push_success:
                print("[Warning] Git push failed. (Data is saved locally)")

            # 少し待機
            time.sleep(2)

    print("\n==================================================")
    print(" All tasks completed successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()