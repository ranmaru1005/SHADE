import numpy as np
import numpy.typing as npt
from scipy.signal import argrelmax, argrelmin

CROSSTALK_PRINT_COUNTER = 0


"""
#宇田川さんのもの
def evaluate_band(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    center_wavelength: float,
    length_of_3db_band: float,
    max_crosstalk: float,
    H_p: float,
    H_s: float,
    H_i: float,
    r_max: float,
    weight: list[float],
    ignore_binary_evaluation: bool = False,
) -> np.float_:
    pass_band, cross_talk = _get_pass_band(x=x, y=y, H_p=H_p, center_wavelength=center_wavelength)
    if pass_band.shape[0] == 1:
        start = pass_band[0][0]
        end = pass_band[0][1]
    # elif cross_talk.shape[0] == 1:
    #     start = cross_talk[0][0]
    #     end = cross_talk[0][1]
    else:
        return np.float_(0)
    result = [
        _evaluate_pass_band(x=x, y=y, H_p=H_p, start=start, end=end),
        _evaluate_stop_band(x=x, y=y, H_p=H_p, H_s=H_s, start=start, end=end),
        _evaluate_insertion_loss(x=x, y=y, H_i=H_i, center_wavelength=center_wavelength),
        _evaluate_3db_band(x=x, y=y, length_of_3db_band=length_of_3db_band, start=start, end=end),
        #_evaluate_ripple(x=x, y=y, r_max=r_max, start=start, end=end, center_wavelength=center_wavelength, length_of_3db_band=length_of_3db_band),
        _evaluate_ripple(x=x, y=y, r_max=r_max, start=start, end=end),
        _evaluate_cross_talk(y=y, max_crosstalk=max_crosstalk, pass_band_start=start, pass_band_end=end),
        _evaluate_shape_factor(x=x, y=y, start=start, end=end),
    ]
    
    #print("_evaluate_pass_band = ",result[0])
    #print("_evaluate_stop_band = ",result[1])
    #print("_evaluate_insertion_loss = ",result[2])
    #print("_evaluate_3db_band = ",result[3])
    #print("_evaluate_ripple = ",result[4])
    #print("_evaluate_cross_talk = ",result[5])
    #print("_evaluate_shape_factor = ",result[6])



    
    n_eval = len(result)
    W_c = weight[:n_eval]
    W_b = weight[n_eval:]
    E_c = np.float_(0)
    E_b = np.float_(1)
    for i in range(n_eval):
        E_c += result[i][0] * W_c[i]
        if not result[i][1]:
            E_b *= W_b[i]
    if ignore_binary_evaluation:
        return E_c
    E = E_c * E_b

    return E
"""



"""
#クロストークのペナルティに傾斜をつけたもの
def evaluate_band(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    center_wavelength: float,
    length_of_3db_band: float,
    max_crosstalk: float,
    H_p: float,
    H_s: float,
    H_i: float,
    r_max: float,
    weight: list[float],
    ignore_binary_evaluation: bool = False,
) -> np.float_:
    pass_band, cross_talk_regions = _get_pass_band(x=x, y=y, H_p=H_p, center_wavelength=center_wavelength)
    if pass_band.shape[0] != 1:
        return np.float_(0)
    
    start = pass_band[0][0]
    end = pass_band[0][1]

    # --- 各指標の評価 ---
    # クロストーク以外の評価
    other_results = [
        _evaluate_pass_band(x=x, y=y, H_p=H_p, start=start, end=end),
        _evaluate_stop_band(x=x, y=y, H_p=H_p, H_s=H_s, start=start, end=end),
        _evaluate_insertion_loss(x=x, y=y, H_i=H_i, center_wavelength=center_wavelength),
        _evaluate_3db_band(x=x, y=y, length_of_3db_band=length_of_3db_band, start=start, end=end),
        _evaluate_ripple(x=x, y=y, r_max=r_max, start=start, end=end),
        _evaluate_shape_factor(x=x, y=y, start=start, end=end),
    ]
    # クロストークの評価
    crosstalk_score, crosstalk_ok, crosstalk_penalty = _evaluate_cross_talk_final(
        y=y, max_crosstalk=max_crosstalk, pass_band_start=start, pass_band_end=end
    )

    # --- 評価スコアの計算 ---
    n_other_eval = len(other_results)
    W_c = weight[:n_other_eval+1] # クロストーク分も含む
    W_b = weight[n_other_eval+1:]

    # E_c の計算
    E_c = np.float_(0)
    for i in range(n_other_eval):
        E_c += other_results[i][0] * W_c[i]
    E_c += crosstalk_score * W_c[n_other_eval] # クロストークの連続スコアもE_cに加算

    if ignore_binary_evaluation:
        return E_c

    # E_b (固定ペナルティ) の計算
    E_b = np.float_(1)
    for i in range(n_other_eval):
        if not other_results[i][1]:
            E_b *= W_b[i]

    # 最終スコアEの計算
    # 固定ペナルティE_bと、クロストークの動的ペナルティの両方を適用
    E = E_c * E_b * crosstalk_penalty

    return E
"""




#クロストークを相対差で評価
def evaluate_band(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    center_wavelength: float,
    length_of_3db_band: float,
    max_crosstalk: float, # ← 指示通り引数名を変更しました
    H_p: float,
    H_s: float,
    H_i: float,
    r_max: float,
    weight: list[float],
    ignore_binary_evaluation: bool = False,
) -> np.float_:
    pass_band, cross_talk_regions = _get_pass_band(x=x, y=y, H_p=H_p, center_wavelength=center_wavelength)
    if pass_band.shape[0] != 1:
        return np.float_(0)
    
    start = pass_band[0][0]
    end = pass_band[0][1]

    main_peak_height = y[start:end].max()

    other_results = [
        _evaluate_pass_band(x=x, y=y, H_p=H_p, start=start, end=end),
        _evaluate_stop_band(x=x, y=y, H_p=H_p, H_s=H_s, start=start, end=end),
        _evaluate_insertion_loss(x=x, y=y, H_i=H_i, center_wavelength=center_wavelength),
        _evaluate_3db_band(x=x, y=y, length_of_3db_band=length_of_3db_band, start=start, end=end),
        _evaluate_ripple(x=x, y=y, r_max=r_max, start=start, end=end),
        _evaluate_shape_factor(x=x, y=y, start=start, end=end),
    ]
    
    # _evaluate_relative_crosstalkに渡す際に、required_crosstalk_dbとして値を渡します
    crosstalk_score, crosstalk_ok, crosstalk_penalty = _evaluate_relative_crosstalk(
        x=x,
        y=y,
        main_peak_height=main_peak_height,
        required_crosstalk_db=max_crosstalk, # ← ここで受け取った値を渡します
        pass_band_start=start,
        pass_band_end=end
    )

    # --- 評価スコアの計算 (変更なし) ---
    n_other_eval = len(other_results)
    W_c = weight[:n_other_eval+1]
    W_b = weight[n_other_eval+1:]

    E_c = np.float_(0)
    for i in range(n_other_eval):
        E_c += other_results[i][0] * W_c[i]
    E_c += crosstalk_score * W_c[n_other_eval]

    if ignore_binary_evaluation:
        return E_c

    E_b = np.float_(1)
    for i in range(n_other_eval):
        if not other_results[i][1]:
            E_b *= W_b[i]

    E = E_c * E_b * crosstalk_penalty

    return E






def _calculate_pass_band_range(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    H_p: float,
) -> npt.NDArray[np.int_]:
    start = 0
    end = x.size - 1
    a: npt.NDArray[np.bool_] = np.where(y <= H_p, True, False)
    b: npt.NDArray[np.bool_] = np.append(a[1:], a[-1])
    pass_band_range: npt.NDArray[np.int_] = np.where(np.logical_xor(a, b))[0]
    if pass_band_range.size == 0:
        return pass_band_range
    if not a[pass_band_range][0]:
        pass_band_range = np.append(start, pass_band_range)
    if a[pass_band_range][-1]:
        pass_band_range = np.append(pass_band_range, end)
    pass_band_range = np.reshape(pass_band_range, [pass_band_range.size // 2, 2])

    return pass_band_range


def _get_pass_band(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    H_p: float,
    center_wavelength: float,
) -> (npt.ArrayLike, npt.ArrayLike):
    pass_band = []
    cross_talk = []
    for start, end in _calculate_pass_band_range(x=x, y=y, H_p=H_p):
        if center_wavelength >= x[start] and center_wavelength <= x[end]:
            pass_band.extend([start, end])
        else:
            cross_talk.extend([start, end])

    pass_band = np.reshape(pass_band, [len(pass_band) // 2, 2])
    cross_talk = np.reshape(cross_talk, [len(cross_talk) // 2, 2])

    return pass_band, cross_talk


def _get_3db_band(x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], start: int, end: int) -> npt.ArrayLike:
    border: np.float_ = y.max() - 3
    a = np.where(y[start:end] <= border, True, False)
    b = np.append(a[1:], a[-1])
    index = np.where(np.logical_xor(a, b))[0]

    return index


def _evaluate_pass_band(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], H_p: float, start: int, end: int
) -> tuple[np.float_, bool]:
    distance: np.float_ = x[1] - x[0]
    a = abs(H_p * (x[end] - x[start]))

    if a == 0:
        return (np.float_(0), False)

    b = abs(np.sum(H_p - y[start:end]) * distance)
    E = b / a

    return (E, True)


def _evaluate_stop_band(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], H_p: float, H_s: float, start: int, end: int
) -> tuple[np.float_, bool]:
    distance: np.float_ = x[1] - x[0]
    c = abs((H_s - H_p) * ((x[start] - x[0]) + (x[-1] - x[end])))

    if c == 0:
        return (np.float_(0), False)

    y1 = np.where(y[0:start] > H_s, H_p - y[0:start], H_p - H_s)
    y1 = np.where(y1 > 0, y1, 0)
    y2 = np.where(y[end:-1] > H_s, H_p - y[end:-1], H_p - H_s)
    y2 = np.where(y2 > 0, y2, 0)
    d = abs((np.sum(y1) + np.sum(y2)) * distance)
    E = d / c

    return (E, True)


def _evaluate_insertion_loss(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    H_i: float,
    center_wavelength: float,
) -> tuple[np.float_, bool]:
    insertion_loss = y[x == center_wavelength]
    if insertion_loss[0] < H_i:
        return (np.float_(0), False)
    E = 1 - insertion_loss[0] / H_i
    return (E, True)


def _evaluate_3db_band(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], length_of_3db_band: float, start: int, end: int
) -> tuple[np.float_, bool]:
    distance: np.float_ = x[1] - x[0]
    index = _get_3db_band(x=x, y=y, start=start, end=end)
    if index.size <= 1:
        return (np.float_(0), False)
    practical_length_of_3db_band = distance * (index[-1] - index[0])
    if practical_length_of_3db_band > length_of_3db_band:
        E = (2 * length_of_3db_band - practical_length_of_3db_band) / length_of_3db_band
    else:
        E = practical_length_of_3db_band / length_of_3db_band
    E = E ** 3
    return (E, True)




#ペナルティ量を変更するためのもの。evaluate_bandにも影響を及ぼす
def _evaluate_cross_talk_final(
    y: npt.NDArray[np.float_],
    max_crosstalk: float,
    pass_band_start: int,
    pass_band_end: int,
    initial_penalty: float = 0.7,
    penalty_rate: float = 2.0
) -> tuple[np.float_, bool, np.float_]:
    """
    クロストークを評価し、違反度合いに応じた動的なペナルティ係数を返す。
    （ペナルティ適用時、100回に1回だけ値を出力する）
    """
    global CROSSTALK_PRINT_COUNTER # グローバル変数のカウンターを使用

    # --- ピーク値の計算 (変更なし) ---
    start_band = y[:pass_band_start]
    end_band = y[pass_band_end:]
    highest_peak = -np.inf
    if start_band.size > 0:
        maxid_start = np.append(argrelmax(start_band)[0], [0, len(start_band) - 1])
        highest_peak = max(highest_peak, start_band[maxid_start].max())
    if end_band.size > 0:
        maxid_end = np.append(argrelmax(end_band)[0], [0, len(end_band) - 1])
        highest_peak = max(highest_peak, end_band[maxid_end].max())
    if np.isneginf(highest_peak):
        return (np.float_(1.0), True, np.float_(1.0))

    # --- 評価値の計算 ---
    is_ok = highest_peak <= max_crosstalk
    dynamic_penalty = np.float_(1.0)

    if is_ok:
        score = np.float_(1.0)
    else:
        score = np.float_(0.0)
        
        # 動的ペナルティを計算
        overage = highest_peak - max_crosstalk
        normalized_rate = penalty_rate / abs(max_crosstalk) if abs(max_crosstalk) > 1e-9 else penalty_rate
        additional_penalty = np.exp(-normalized_rate * overage)
        dynamic_penalty = initial_penalty * additional_penalty
        
        # --- ここからが修正部分 ---
        CROSSTALK_PRINT_COUNTER += 1 # カウンターを1増やす
        if CROSSTALK_PRINT_COUNTER % 100 == 0: # 100で割り切れる時だけ表示
            print(f"[Crosstalk Penalty] Peak:{highest_peak:.2f} > Threshold:{max_crosstalk:.2f} -> Penalty:{dynamic_penalty:.3f}")
        # --- ここまでが修正部分 ---

    return (score, is_ok, dynamic_penalty)








"""
#3dB波長帯域の選択を場合で分ける
def _evaluate_ripple(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    r_max: float = 1.0,
    start: int = 0,
    end: int = -1,
    center_wavelength: float = None,
    length_of_3db_band: float = None
) -> tuple[np.float_, bool]:
    if end == -1:
        end = len(x)

    index = _get_3db_band(x=x, y=y, start=start, end=end)

    if index.size < 2:
        #print("3dB帯域のインデックスが見つかりません")
        return (np.float_(0), False)

    idx_start = start + index[0]
    idx_end = start + index[1]

    if idx_end > len(x):
        return (np.float_(0), False)

    # 実測3dB波長帯域の範囲
    actual_range = x[idx_end] - x[idx_start]

    use_fixed_band = True
    if center_wavelength is not None and length_of_3db_band is not None:
        expected_range = length_of_3db_band
        lower_bound = 0.9 * expected_range
        upper_bound = 1.1 * expected_range

        if lower_bound <= actual_range <= upper_bound:
            use_fixed_band = False  # 実測3dB帯域を使う

    # 波長範囲とスペクトル取得
    if use_fixed_band:
        # 中心波長 ± length_of_3db_band/2 を使用
        lower = center_wavelength - length_of_3db_band / 2
        upper = center_wavelength + length_of_3db_band / 2
        indices = np.where((x >= lower) & (x <= upper))[0]
        #print(f"固定帯域を使用: {lower*1e9:.3f} nm ～ {upper*1e9:.3f} nm")
    else:
        indices = np.arange(idx_start, idx_end)
        #print(f"実測帯域を使用: {x[idx_start]*1e9:.3f} nm ～ {x[idx_end]*1e9:.3f} nm")

    if indices.size < 2:
        #print("評価対象のデータが不足しています")
        return (np.float_(0), False)

    band_y = y[indices]
    std = np.std(band_y)

    #print(f"標準偏差（ripple）= {std:.4f} dB")

    if std > r_max:
        return (np.float_(0), False)

    score = 1 - std / r_max
    return (np.float_(score), True)

"""


"""
#3dB波長帯域を中心波長から1nmで固定
def _evaluate_ripple(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    r_max: float,
    start: int,
    end: int,
    center_wavelength: float = None,
    length_of_3db_band: float = None
) -> tuple[np.float_, bool]:
    # 中心波長と評価幅が与えられている場合はそれを使用
    if center_wavelength is not None and length_of_3db_band is not None:
        half_band = length_of_3db_band / 2
        lower = center_wavelength - half_band
        upper = center_wavelength + half_band

        # 評価範囲内のインデックス取得
        band_indices = np.where((x >= lower) & (x <= upper))[0]

        if band_indices.size < 2:
            print("指定した中心波長と3dB帯域幅で十分なデータがありません。")
            return (np.float_(0), False)

        band_y = y[band_indices]
        std = np.std(band_y)

        print(f"評価範囲: {lower*1e9:.3f} nm ～ {upper*1e9:.3f} nm")
        print(f"標準偏差 (ripple): {std:.4f}")

        if std > r_max:
            return (np.float_(0), False)

        score = 1 - std / r_max
        return (np.float_(score), True)

    else:
        # fallback: 元の方式で3dB帯域を使ったリップル評価
        pass_band = y[start:end]
        index = _get_3db_band(x=x, y=y, start=start, end=end)
        if index.size <= 1:
            return (np.float_(0), False)
        three_db_band = pass_band[index[0]: index[-1]]
        std = np.std(three_db_band)

        if std > r_max:
            return (np.float_(0), False)
        score = 1 - std / r_max
        return (np.float_(score), True)
"""



"""
#標準偏差を用いる
def _evaluate_ripple(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    r_max: float = 1.0,
    start: int = 0,
    end: int = -1
) -> tuple[np.float_, bool]:
    if end == -1:
        end = len(x)

    index = _get_3db_band(x=x, y=y, start=start, end=end)

    if index.size < 2:
        print("3dB帯域のインデックスが見つかりません")
        return (np.float_(0), False)

    idx_start = start + index[0]
    idx_end = start + index[1]

    if idx_end > len(y):
        return (np.float_(0), False)

    # 波長範囲表示
    start_wavelength = x[idx_start] * 1e9
    end_wavelength = x[idx_end] * 1e9
    print(f"3dB波長帯域の範囲: {start_wavelength:.3f} nm ～ {end_wavelength:.3f} nm")

    # スペクトルデータ抽出
    three_db_band = y[idx_start:idx_end]

    # 標準偏差でばらつきを評価
    std = np.std(three_db_band)

    if std > r_max:
        return (np.float_(0), False)

    score = 1 - std / r_max
    return (score, True)
"""



def _evaluate_ripple(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], r_max: float, start: int, end: int
) -> tuple[np.float_, bool]:
    pass_band = y[start:end]
    index = _get_3db_band(x=x, y=y, start=start, end=end)
    if index.size <= 1:
        return (np.float_(0), False)
    idx_start = start + index[0]
    idx_end = start + index[-1]
    start_wavelength = x[idx_start] * 1e9
    end_wavelength = x[idx_end] * 1e9
    #print(f"3dB波長帯域の範囲: {start_wavelength:.3f} nm ～ {end_wavelength:.3f} nm")
    three_db_band = pass_band[index[0] : index[-1]]
    maxid = argrelmax(three_db_band, order=1)
    minid = argrelmin(three_db_band, order=1)
    peak_max = three_db_band[maxid]
    peak_min = three_db_band[minid]
    if len(peak_min) == 0:
        return (1, True)
    dif = peak_max.max() - peak_min.min()
    if dif > r_max:
        return (np.float_(0), False)
    E = 1 - dif / r_max
    return (E, True)



"""
#指数的に評価を下げるクロストーク
def _evaluate_cross_talk(
    y: npt.NDArray[np.float_],
    max_crosstalk: float,
    pass_band_start: int,
    pass_band_end: int,
    alpha: float = 2.0  # 減衰の強さ
) -> tuple[np.float_, bool]:
    # 前後ストップバンドのスペクトル抽出
    start_band = y[:pass_band_start]
    end_band = y[pass_band_end:]

    # 局所最大値のインデックス
    maxid_start = np.append(0, argrelmax(start_band)[0])
    maxid_end = np.append(argrelmax(end_band)[0], -1)

    # ピーク値取得（両端で最大のものを採用）
    start_peak = start_band[maxid_start].max() if maxid_start.size > 0 else 0
    end_peak = end_band[maxid_end].max() if maxid_end.size > 0 else 0
    peak = max(start_peak, end_peak)

    #print(f"クロストーク最大ピーク: {peak:.3f} dB")

    # スコア計算（指数型）
    normalized = peak / max_crosstalk
    score = np.exp(-alpha * normalized)
    score = float(np.clip(score, 0.0, 1.0))  # 念のため0〜1に制限

    return (np.float_(score), True)
"""


"""
#クロストークの制限を取り払ったもの
def _evaluate_cross_talk(
    y: npt.NDArray[np.float_],
    max_crosstalk: float,
    pass_band_start: int,
    pass_band_end: int
) -> tuple[np.float_, bool]:
    # ストップバンドの前後データを抽出
    start_band = y[:pass_band_start]
    end_band = y[pass_band_end:]

    # 局所的なピーク検出
    maxid_start = np.append(0, argrelmax(start_band)[0])
    maxid_end = np.append(argrelmax(end_band)[0], -1)

    # 各ストップバンドにおける最大ピーク値取得
    start_peak = start_band[maxid_start].max() if maxid_start.size > 0 else 0
    end_peak = end_band[maxid_end].max() if maxid_end.size > 0 else 0

    # クロストーク評価用ピーク値（左右のうち大きい方）
    peak = max(start_peak, end_peak)

    #print(f"クロストーク最大ピーク: {peak:.3f} dB")

    # スコア計算（必ず 0〜1 の範囲に収める）
    score = max(0.0, 1.0 - (peak / max_crosstalk))

    return (np.float_(score), True)
"""




"""
def _evaluate_cross_talk(
    y: npt.NDArray[np.float_], max_crosstalk: float, pass_band_start: int, pass_band_end: int
) -> tuple[np.float_, bool]:
    start = y[:pass_band_start]
    end = y[pass_band_end:]
    maxid_start = np.append(0, argrelmax(start))
    maxid_end = np.append(argrelmax(end), -1)
    start_peak = start[maxid_start]
    end_peak = end[maxid_end]
    a = np.any(start_peak > max_crosstalk)
    b = np.any(end_peak > max_crosstalk)
    if a or b:
        return (np.float_(0), False)
    return (np.float_(0), True)
"""


#クロストークの評価をピークとサイドピークの差で行う
def _evaluate_relative_crosstalk(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    main_peak_height: float,
    required_crosstalk_db: float,
    pass_band_start: int,
    pass_band_end: int,
    initial_penalty: float = 0.9,
    penalty_rate: float = 2.0
) -> tuple[np.float_, bool, np.float_]:
    """
    メインピークとの相対差でクロストークを評価し、表示量を調整したデバッグ情報を出力する。
    """
    global CROSSTALK_PRINT_COUNTER # グローバル変数のカウンターを使用

    # --- サイドピーク領域で最も高いピークとその波長を探す (変更なし) ---
    side_band_start_y = y[:pass_band_start]
    side_band_end_y = y[pass_band_end:]
    
    highest_side_peak_y = -np.inf
    highest_side_peak_x = 0.0

    if side_band_start_y.size > 0:
        peak_y = np.max(side_band_start_y)
        if peak_y > highest_side_peak_y:
            highest_side_peak_y = peak_y
            idx = np.argmax(side_band_start_y)
            highest_side_peak_x = x[:pass_band_start][idx]

    if side_band_end_y.size > 0:
        peak_y = np.max(side_band_end_y)
        if peak_y > highest_side_peak_y:
            highest_side_peak_y = peak_y
            idx = np.argmax(side_band_end_y)
            highest_side_peak_x = x[pass_band_end:][idx]
    
    if np.isneginf(highest_side_peak_y):
        return (np.float_(1.0), True, np.float_(1.0))

    # --- 評価値の計算 (変更なし) ---
    actual_crosstalk = main_peak_height - highest_side_peak_y
    is_ok = actual_crosstalk >= required_crosstalk_db
    dynamic_penalty = np.float_(1.0)

    if is_ok:
        score = np.float_(1.0)
    else:
        score = np.float_(0.0)
        shortage = required_crosstalk_db - actual_crosstalk
        additional_penalty = np.exp(-penalty_rate * shortage)
        dynamic_penalty = initial_penalty * additional_penalty

        # --- ここからが表示量調整部分 ---
        CROSSTALK_PRINT_COUNTER += 1 # ペナルティ発生時のみカウンターを増やす
        if CROSSTALK_PRINT_COUNTER % 100 == 0: # 100回に1回だけ表示
            print("-----------------------------------------")
            print(f"Crosstalk Violation (Count: {CROSSTALK_PRINT_COUNTER}):")
            print(f"  - Side Peak Wavelength: {highest_side_peak_x * 1e9:.3f} nm")
            print(f"  - Calculated Crosstalk: {actual_crosstalk:.2f} dB (Required: {required_crosstalk_db:.2f} dB)")
            print(f"  => Final Penalty     : {dynamic_penalty:.4f}")
            print("-----------------------------------------")
        # --- ここまでが表示量調整部分 ---

    return (score, is_ok, dynamic_penalty)




def _evaluate_shape_factor(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], start: int, end: int
) -> tuple[np.float_, bool]:
    index = _get_3db_band(x=x, y=y, start=start, end=end)
    if index.size <= 1:
        return (np.float_(0), False)
    E = (index[-1] - index[0]) / (end - start)
    if E < 0.5:
        return (E, False)
    return (E, True)
