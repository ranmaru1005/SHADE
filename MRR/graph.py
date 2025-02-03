from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import rc
from matplotlib.ticker import AutoLocator, FormatStrFormatter, MultipleLocator

rc("text", usetex=True)
rc("font", size=16)


class Graph:
    def __init__(self, is_focus: bool = False):
        self.is_focus = is_focus

    def create(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

    def plot(
        self,
        x: Union[npt.NDArray[np.float_], list],
        y: Union[npt.NDArray[np.float_], list],
        label: Optional[str] = None,
    ) -> None:
        """ x, y がリストでも NumPy 配列に変換してプロット """

        # 🔹 NumPy 配列に変換
        x = np.array(x) if not isinstance(x, np.ndarray) else x
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        # 🔹 x 軸の単位を nm に変換
        self.ax.semilogx(x * 1e9, y, label=label)

    def show(
        self,
        img_path: Path = Path("img/out.pdf"),
    ) -> None:
        """ グラフの表示と保存 """

        # 🔹 軸ラベル設定
        self.ax.set_xlabel(r"Wavelength $\lambda$ (nm)", fontsize=24)
        self.ax.set_ylabel("Transmittance (dB)", fontsize=24)

        # 🔹 軸範囲の設定
        self.ax.set_xlim([1530, 1570])  # 🔹 x 軸: 1530 ~ 1570 nm
        self.ax.set_ylim([-60, 5])  # 🔹 y 軸: -60 ~ 5 dB

        # 🔹 軸目盛りの設定
        self.ax.xaxis.set_major_locator(MultipleLocator(10))  # 10 nm ごとに目盛り
        self.ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        self.ax.xaxis.set_minor_locator(MultipleLocator(5))  # 5 nm の補助目盛り
        self.ax.yaxis.set_major_locator(MultipleLocator(10))  # 10 dB ごとに目盛り
        self.ax.yaxis.set_minor_locator(MultipleLocator(5))  # 5 dB の補助目盛り

        # 🔹 判例を設定
        plt.legend(loc="upper center", fontsize=12, frameon=False)

        # 🔹 グラフ保存
        self.fig.savefig(img_path)
        plt.show()
