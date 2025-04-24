"""
特異値分解 (SVD) を用いた画像圧縮と再構成を可視化するGradioアプリケーション。

指定されたURLから画像を読み込み、グレイスケールに変換後、
特異値分解 (U, S, Vt) を行います。
ユーザーはスライダーを操作して、再構成に使用する特異値の数（ランク）を選択でき、
それによって画像がどのように再構成されるか、また分解された各行列（U, Sigma, Vt）の
性質を視覚的に確認できます。
"""

from io import BytesIO

import gradio as gr
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401 日本語表示用
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.cm import ScalarMappable
from PIL import Image

# 画像URLのリスト
example_urls = [
    [
        "単純な斜めの縞模様",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSoki9ZKPtsoYw1z_A2DZQIpMnObustPL7OKA&s",
    ],
    [
        "少し複雑な斜めの縞模様",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRleTnZSi7n7K28PycQ4IduBkFTn3qyd5t0QA&s",
    ],
    [
        "二頭のシマウマ",
        "https://images.rawpixel.com/image_800/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L2lzODk3OS1pbWFnZS1rd3Z5ZW5sZi5qcGc.jpg",
    ],
    [
        "シマウマのアップ",
        "https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvcHgxMjU3NDI0LWltYWdlLWpvYjYzMC1uLWwwZzA4Zzc2LmpwZw.jpg",
    ],
    [
        "規則正しい縦縞",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR5dXqHnnrYPqQNSrEWj6_TH3rLEF1FcCCatsAm6pcx_crAiNmWl2pdfTXNNe04_JZMnio&usqp=CAU",
    ],
    [
        "川と空港",
        "https://live.staticflickr.com/7497/15364254203_e6d7a2465b_b.jpg",
    ],
]

# Wikipedia リンク
wikipedia_link = "https://ja.wikipedia.org/wiki/%E7%89%B9%E7%95%B0%E5%80%A4%E5%88%86%E8%A7%A3"

GUIDE_TEXT = f"""
# 特異値分解 (SVD) 可視化ツール ➗

このツールは、行列を分解する強力な手法の一つである
[**特異値分解 (Singular Value Decomposition, SVD)**]({wikipedia_link})
を、画像の圧縮と再構成を通じて直感的に理解するためのものです。

グレイスケール画像を数学的な「行列」とみなし、SVD によって３つの行列 (U, Σ, V<sup>T</sup>) に分解します。
特に、Σ (シグマ) の対角成分である **特異値** は、元の画像の情報をどれだけ含んでいるかの指標となります。
"""

USAGE_TEXT = """
1.  **画像の指定:**
    * 下の「サンプル画像のURL」から一つ選んでクリックするか、
    * 「1. 画像URLを入力」欄に直接画像のURLを貼り付けてください。
    * URLを入力したら、「実行」ボタンを押すか、Enterキーを押します。
2.  **分解結果の確認:**
    * 画像が読み込まれると、元のグレイスケール画像と、SVDで分解された行列 (U, Σ, V<sup>T</sup>) の情報が下に表示されます。
    * 「完全に再構成された画像」は、すべての特異値を使って元画像を復元したものです。通常、元のグレイスケール画像とほぼ同じになります。
    * 「分解された行列の分析」セクションで、各行列のヒートマップや特異値の分布を確認できます。
3.  **特異値の数を変えて再構成:**
    * 「2. 再構成に使う特異値のインデックス $k$」のスライダーを動かしてみてください。
    * スライダーで選択したインデックス $k$ に応じて、下の３つの再構成画像が変わります。
        * **上位 $k+1$ 個の特異値で再構成:** インデックス $0$ から $k$ までの、より大きな特異値を使って再構成した画像です。少ない特異値でも画像の主要な特徴が捉えられていることがわかります。
        * **インデックス $k$ の特異値のみで再構成:** 選択した $k$ 番目の特異値一つだけを使って画像を構成する要素です。各特異値が画像のどのパターンに対応するかが見えます。
        * **インデックス $k+1$ 以降の特異値で再構成:** $k$ 番目より小さい（情報の寄与が少ない）特異値だけで再構成した画像です。ノイズのような細かい成分に対応することが多いです。
    * 「メモ」欄には、選択した $k$ に関する詳細情報や、データ圧縮率の目安が表示されます。
"""


# --- Matplotlib Plotting Functions ---


def plot_matrix_heatmap(matrix_data, title):
    """
    行列データをヒートマップと値の分布ヒストグラムで可視化する。

    Args:
        matrix_data (np.ndarray): 可視化する行列データ。
        title (str): グラフのタイトル。

    Returns:
        matplotlib.figure.Figure: 生成されたグラフのFigureオブジェクト。
    """
    plt.close("all")  # 既存のプロットを閉じる

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=16)  # 全体のタイトル

    # --- ヒートマップ ---
    ax_heatmap = axes[0]
    cmap = plt.get_cmap("viridis")  # カラーマップ
    # データ範囲に基づいて正規化
    norm = plt.Normalize(vmin=np.min(matrix_data), vmax=np.max(matrix_data))
    ax_heatmap.imshow(matrix_data, cmap=cmap, norm=norm, aspect="auto")
    ax_heatmap.set_title("ヒートマップ")
    ax_heatmap.set_xlabel("列インデックス")
    ax_heatmap.set_ylabel("行インデックス")
    # カラーバーを追加
    fig.colorbar(
        ScalarMappable(cmap=cmap, norm=norm), ax=ax_heatmap, shrink=0.8
    )

    # --- ヒストグラム ---
    ax_hist = axes[1]
    sns.histplot(matrix_data.flatten(), kde=True, ax=ax_hist, bins=30)
    ax_hist.set_title("値の分布")
    ax_hist.set_xlabel("値")
    ax_hist.set_ylabel("頻度")
    ax_hist.grid(True, axis="y", linestyle="--", alpha=0.7)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # タイトルとの重なりを避ける
    return fig


def plot_singular_values(singular_values):
    """
    特異値とその累積寄与率をプロットする。

    Args:
        singular_values (np.ndarray): 特異値の配列 (降順ソート済み)。

    Returns:
        matplotlib.figure.Figure: 生成されたグラフのFigureオブジェクト。
    """
    plt.close("all")  # 既存のプロットを閉じる

    if singular_values is None or len(singular_values) == 0:
        # データがない場合は空のグラフを返すか、エラーメッセージを表示
        fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle("特異値プロット (データなし)", fontsize=16)
        ax[0].set_title("特異値 (線形スケール)")
        ax[1].set_title("特異値 (対数スケール)")
        ax[2].set_title("累積寄与率")
        ax[2].set_xlabel("特異値のインデックス (大きい順)")
        for a in ax:
            a.grid(True, linestyle="--", alpha=0.7)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    indices = np.arange(len(singular_values))

    # 累積寄与率 (分散ベース)
    # 特異値の二乗はデータの分散に対応するため、二乗和の累積割合を計算
    variance_explained = singular_values**2
    cumulative_contribution_ratio = np.cumsum(variance_explained) / np.sum(
        variance_explained
    )

    # --- グラフ描画 ---
    # 3つのグラフを縦に並べる (線形スケール、対数スケール、累積寄与率)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("特異値の分布と累積寄与率", fontsize=16)

    # 1. 線形スケールでの特異値プロット
    axes[0].plot(indices, singular_values, marker=None, linestyle="-")
    axes[0].set_title("特異値 (線形スケール)")
    axes[0].set_ylabel("特異値")
    axes[0].grid(True, linestyle="--", alpha=0.7)

    # 2. 対数スケールでの特異値プロット
    axes[1].plot(indices, singular_values, marker=None, linestyle="-")
    axes[1].set_yscale("log")  # Y軸を対数スケールに
    axes[1].set_title("特異値 (対数スケール)")
    axes[1].set_ylabel("特異値 (対数)")
    axes[1].grid(True, linestyle="--", alpha=0.7)

    # 3. 累積寄与率プロット
    axes[2].plot(
        indices, cumulative_contribution_ratio, marker=None, linestyle="-"
    )
    axes[2].set_title("累積寄与率（特異値の二乗和ベース）")
    axes[2].set_xlabel("特異値のインデックス (大きい順)")
    axes[2].set_ylabel("累積寄与率")
    axes[2].set_ylim(0, 1.05)  # Y軸の範囲を0から1.05に設定
    axes[2].grid(True, linestyle="--", alpha=0.7)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # タイトルとの重なりを避ける
    return fig


# --- Image Processing and SVD Functions ---


def reconstruct_image(U, S_vector, Vt, k=None):
    """
    特異値分解の結果 (U, S, Vt) から画像を再構成する。
    オプションで指定されたインデックス k までの特異値を使用する。

    Args:
        U (np.ndarray): 左特異ベクトル行列。
        S_vector (np.ndarray): 特異値ベクトル (1次元配列)。
        Vt (np.ndarray): 右特異ベクトル行列 (転置済み)。
        k (int, optional): 使用する特異値の数 (上位 k 個)。
                             None の場合は全ての特異値を使用する。

    Returns:
        PIL.Image.Image: 再構成された画像の PIL Image オブジェクト。
                         入力が無効な場合はNoneを返す。
    """
    if U is None or S_vector is None or Vt is None:
        print("エラー: U, S, Vt のいずれかが None です。再構成できません。")
        return None

    # 有効な k の値を決定
    max_k = len(S_vector)
    if k is None:
        effective_k = max_k
    else:
        effective_k = min(max(0, k), max_k)  # 0 <= effective_k <= max_k

    # 特異値ベクトルから対角行列 Sigma を作成
    # 元の行列サイズに合わせて Sigma を作成する必要がある
    num_rows = U.shape[0]
    num_cols = Vt.shape[1]
    Sigma_matrix = np.zeros((num_rows, num_cols))

    # 上位 k 個の特異値のみを対角成分にセット
    # min(num_rows, num_cols) と effective_k の小さい方まで
    diag_len = min(num_rows, num_cols, effective_k)
    if diag_len > 0:
        np.fill_diagonal(
            Sigma_matrix[:diag_len, :diag_len], S_vector[:diag_len]
        )

    # 再構成: reconstructed = U @ Sigma @ Vt
    # スライスを使わずに U, Vt 全体を使うことで、正しい次元で計算される
    reconstructed_array = U @ Sigma_matrix @ Vt

    # 値を [0, 1] から [0, 255] の範囲にスケールし、整数型に変換
    # clip で範囲外の値を取り除く
    reconstructed_array_uint8 = (
        (reconstructed_array * 255.0).clip(0, 255).astype(np.uint8)
    )

    # NumPy 配列から PIL Image オブジェクトに変換
    reconstructed_image_pil = Image.fromarray(reconstructed_array_uint8)

    return reconstructed_image_pil


def load_image_and_perform_svd(image_url):
    """
    指定されたURLから画像を読み込み、グレイスケール変換し、SVDを実行する。

    Args:
        image_url (str): 画像のURL。

    Returns:
        tuple: 以下の要素を含むタプル:
            - original_image (PIL.Image.Image): 読み込まれた元のグレイスケール画像。
            - svd_results (dict): 'U', 'S', 'Vt' をキーとするSVD結果の辞書。
                                  読み込み失敗時は None を返す。
            - error_message (str or None): エラーが発生した場合のエラーメッセージ。
    """
    try:
        # URLから画像データを取得
        response = requests.get(image_url)
        response.raise_for_status()  # HTTPエラーチェック

        # BytesIO を使ってメモリ上で画像を処理し、PIL Image オブジェクトに変換
        # 'L' モードでグレイスケールに変換
        img_pil = Image.open(BytesIO(response.content)).convert("L")

        # PIL Image を NumPy 配列に変換し、値を [0, 1] の範囲に正規化
        img_array_normalized = np.array(img_pil) / 255.0

        # SVD (特異値分解) を実行
        # U: 左特異ベクトル (m x m) または (m x k)
        # S: 特異値 (1次元配列, 降順)
        # Vt: 右特異ベクトルの転置 (n x n) または (k x n)
        # full_matrices=False にすると、 U (m x k), Vt (k x n) となり、
        # k = min(m, n) となるため、計算効率が良い場合がある。
        U, S, Vt = np.linalg.svd(img_array_normalized, full_matrices=True)

        svd_results = {"U": U, "S": S, "Vt": Vt}
        return img_pil, svd_results, None  # 成功時はエラーメッセージなし

    except requests.exceptions.RequestException as e:
        error_msg = f"画像URLへのアクセスに失敗しました: {e}"
        print(error_msg)
        return None, None, error_msg
    except IOError as e:
        error_msg = f"画像の読み込みまたは形式の変換に失敗しました: {e}"
        print(error_msg)
        return None, None, error_msg
    except np.linalg.LinAlgError as e:
        error_msg = f"SVD計算に失敗しました: {e}"
        print(error_msg)
        return None, None, error_msg
    except Exception as e:
        # 予期せぬエラー
        error_msg = f"予期せぬエラーが発生しました: {e}"
        print(error_msg)
        return None, None, error_msg


def create_diagnostic_plots(svd_results):
    """
    SVDの結果 (U, S, Vt) を用いて診断プロットを作成する。

    Args:
        svd_results (dict): 'U', 'S', 'Vt' をキーとするSVD結果の辞書。

    Returns:
        tuple: 以下の Matplotlib Figure オブジェクトを含むタプル:
            - U_plot (Figure): U行列のヒートマップと分布。
            - S_plot (Figure): S (特異値) のヒートマップ風表示と分布 (対数スケール)。
            - S_dist_plot (Figure): S (特異値) の分布と累積寄与率プロット。
            - Vt_plot (Figure): Vt行列のヒートマップと分布。
    """
    if svd_results is None:
        # SVD結果がない場合は空のプロットを返す
        empty_fig = plt.figure()
        return empty_fig, empty_fig, empty_fig, empty_fig

    U = svd_results["U"]
    S = svd_results["S"]
    Vt = svd_results["Vt"]

    # --- U 行列のプロット ---
    U_plot = plot_matrix_heatmap(U, "左特異ベクトル行列 U")

    # --- S (特異値) のプロット ---
    # Sはベクトルなので、ヒートマップ風に見せるためにタイル表示する
    # 対数スケールで表示すると値の差が見やすい場合がある
    S_log10 = np.log10(S + 1e-10)  # log(0) を避けるため微小値を追加
    # 横幅を U の列数 (または Vt の行数) に合わせる
    num_cols_s_map = min(U.shape[1], Vt.shape[0])
    S_heatmap_data = np.tile(S_log10[:, np.newaxis], (1, num_cols_s_map))
    S_plot = plot_matrix_heatmap(
        S_heatmap_data, "特異値ベクトル S (log10スケールでタイル表示)"
    )

    # --- S (特異値) の分布プロット ---
    S_dist_plot = plot_singular_values(S)

    # --- Vt 行列のプロット ---
    Vt_plot = plot_matrix_heatmap(Vt, "右特異ベクトル行列 Vt")

    return U_plot, S_plot, S_dist_plot, Vt_plot


def create_reconstruction_details(svd_results, k_index):
    """
    指定されたインデックス k に基づいて、3種類の再構成画像と
    関連情報（ヒートマップ、統計情報）を生成する。

    Args:
        svd_results (dict): 'U', 'S', 'Vt' をキーとするSVD結果の辞書。
        k_index (int): 注目する特異値のインデックス (0から始まる)。

    Returns:
        tuple: 以下の要素を含むタプル:
            - img_higher (PIL.Image): kより大きい特異値を使った再構成画像。
            - img_single (PIL.Image): k番目の特異値のみを使った再構成画像。
            - img_lower (PIL.Image): kより小さい特異値を使った再構成画像。
            - plot_higher (Figure): img_higher のヒートマップ。
            - plot_single (Figure): img_single のヒートマップ。
            - plot_lower (Figure): img_lower のヒートマップ。
            - memo_text (str): 圧縮率などの情報テキスト。
            Noneの場合、入力が無効であることを示す。
    """
    if svd_results is None:
        # SVD結果がない場合は何も返さない (または空の要素)
        empty_img = Image.new("L", (50, 50), color="gray")  # ダミー画像
        empty_fig = plt.figure()
        return (
            empty_img,
            empty_img,
            empty_img,
            empty_fig,
            empty_fig,
            empty_fig,
            "SVD結果がありません。",
        )

    U = svd_results["U"]
    S = svd_results["S"]
    Vt = svd_results["Vt"]

    num_singular_values = len(S)
    # k_index が有効な範囲内か確認
    if not (0 <= k_index < num_singular_values):
        k_index = max(0, min(k_index, num_singular_values - 1))
        print(
            f"警告: 指定されたインデックス {k_index} は範囲外です。{k_index} に調整しました。"
        )

    # --- 再構成画像の生成 ---
    # 1. 上位 k+1 個 (インデックス 0 から k まで) の特異値で再構成
    img_higher = reconstruct_image(U, S, Vt, k=k_index + 1)  # k+1 を指定

    # 2. k 番目の特異値 *のみ* で再構成
    S_single = np.zeros_like(S)
    if k_index < num_singular_values:
        S_single[k_index] = S[k_index]
    img_single = reconstruct_image(
        U, S_single, Vt
    )  # 全ての要素を使うが、k番目以外は0

    # 3. インデックス k+1 以降 (k番目より小さい) の特異値で再構成
    S_lower = np.copy(S)
    S_lower[: k_index + 1] = 0  # 上位 k+1 個をゼロにする
    img_lower = reconstruct_image(U, S_lower, Vt)

    # --- ヒートマップの生成 ---
    plot_higher = (
        plot_matrix_heatmap(
            np.array(img_higher) / 255.0,
            f"上位 {k_index+1} 個の特異値による再構成",
        )
        if img_higher
        else plt.figure()
    )
    plot_single = (
        plot_matrix_heatmap(
            np.array(img_single) / 255.0,
            f"インデックス {k_index} の特異値のみによる再構成",
        )
        if img_single
        else plt.figure()
    )
    plot_lower = (
        plot_matrix_heatmap(
            np.array(img_lower) / 255.0,
            f"インデックス {k_index+1} 以降の特異値による再構成",
        )
        if img_lower
        else plt.figure()
    )

    # --- 統計情報の作成 ---
    k = k_index + 1  # 使用する特異値の数 (ランク)
    original_rows, original_cols = U.shape[0], Vt.shape[1]

    # 元のデータ量 (m*n)
    original_elements = original_rows * original_cols

    # 圧縮後のデータ量 (Uのk列 + Sのk個 + Vtのk行)
    # U[:, :k] は (m, k)
    # S[:k] は (k,)
    # Vt[:k, :] は (k, n)
    compressed_elements = (U.shape[0] * k) + k + (k * Vt.shape[1])

    # SVD分解そのものに必要なデータ量 (U全体 + S全体 + Vt全体)
    # full_matrices=True の場合: m*m + min(m,n) + n*n
    # full_matrices=False の場合: m*k' + k' + k'*n (k' = min(m,n))
    # ここでは単純化のため U, S, Vt の実際の要素数を数える
    svd_storage = U.size + S.size + Vt.size

    # --- メモ文字列の作成 ---
    memo_lines = []
    memo_lines.append(
        f"**画像サイズ:** {original_rows} x {original_cols} ピクセル"
    )
    memo_lines.append(f"**特異値の総数:** {num_singular_values}")
    memo_lines.append(f"**選択されたインデックス k:** {k_index} (0から始まる)")
    memo_lines.append(f"**再構成に使用した特異値の数 (ランク):** {k}")
    memo_lines.append("-" * 20)
    memo_lines.append("**特異値 S の統計情報:**")
    try:
        # pandas Series を使って記述統計量を計算
        s_series = pd.Series(S)
        memo_lines.append(f"  - 平均: {s_series.mean():.4f}")
        memo_lines.append(f"  - 標準偏差: {s_series.std():.4f}")
        memo_lines.append(f"  - 最小値: {s_series.min():.4f}")
        memo_lines.append(
            f"  - 25パーセンタイル: {s_series.quantile(0.25):.4f}"
        )
        memo_lines.append(
            f"  - 中央値 (50パーセンタイル): {s_series.median():.4f}"
        )
        memo_lines.append(
            f"  - 75パーセンタイル: {s_series.quantile(0.75):.4f}"
        )
        memo_lines.append(f"  - 最大値: {s_series.max():.4f}")
    except Exception as e:
        memo_lines.append(f"  - 統計情報の計算エラー: {e}")
    memo_lines.append("-" * 20)
    memo_lines.append(f"**選択されたインデックス {k_index} の情報:**")
    memo_lines.append(f"  - 特異値 S[{k_index}]: {S[k_index]:.4f}")
    # ベクトルの表示は長くなる可能性があるので、形状と最初の数要素だけ表示
    memo_lines.append(
        f"  - 左特異ベクトル U[:, {k_index}] (形状: {U[:, k_index].shape}): "
        f"[{', '.join(f'{x:.2f}' for x in U[:3, k_index])}, ...]"
    )
    memo_lines.append(
        f"  - 右特異ベクトル Vt[{k_index}, :] (形状: {Vt[k_index, :].shape}): "
        f"[{', '.join(f'{x:.2f}' for x in Vt[k_index, :3])}, ...]"
    )
    memo_lines.append("-" * 20)
    memo_lines.append("**データ量比較 (要素数ベース):**")
    memo_lines.append(f"  - 元画像の要素数 (m * n): {original_elements}")
    memo_lines.append(f"  - SVDに必要な要素数 (U + S + Vt): {svd_storage}")
    memo_lines.append(
        f"  - ランク {k} で圧縮した場合の要素数 (m*k + k + k*n): "
        f"{compressed_elements}"
    )
    if original_elements > 0:
        compression_ratio_vs_original = compressed_elements / original_elements
        compression_ratio_vs_svd = (
            compressed_elements / svd_storage if svd_storage > 0 else 0
        )
        memo_lines.append(
            f"  - 圧縮率 (vs 元画像): {compression_ratio_vs_original:.4f} "
            f"({compression_ratio_vs_original*100:.2f}%)"
        )
        if svd_storage > 0:
            memo_lines.append(
                f"  - 圧縮率 (vs SVD全体): {compression_ratio_vs_svd:.4f} "
                f"({compression_ratio_vs_svd*100:.2f}%)"
            )

    memo_text = "\n\n".join(memo_lines)

    # None チェックを追加
    if img_higher is None:
        img_higher = Image.new("L", (50, 50), color="lightgray")
    if img_single is None:
        img_single = Image.new("L", (50, 50), color="lightgray")
    if img_lower is None:
        img_lower = Image.new("L", (50, 50), color="lightgray")

    return (
        img_higher,
        img_single,
        img_lower,
        plot_higher,
        plot_single,
        plot_lower,
        memo_text,
    )


# --- Gradio Application Setup ---

# --- Gradio Interface Definition ---
with gr.Blocks(
    theme=gr.themes.Monochrome(),
    title="特異値分解 可視化ツール",
    fill_width=True,
) as demo:
    # --- 1. 説明 ---
    gr.Markdown(GUIDE_TEXT)

    with gr.Accordion(label="ツールの使い方", open=False):
        gr.Markdown(USAGE_TEXT)

    # --- 2. 入力セクション ---
    with gr.Row():
        url_label = gr.Textbox(
            scale=1,  # 横幅の比率
            label="画像タイトル",
        )
        url_input = gr.Textbox(
            lines=1,
            placeholder="画像のURLを入力してください...",
            scale=4,  # 横幅の比率
            label="1. 画像URLを入力",
        )
        submit_button = gr.Button(value="実行", scale=1, variant="primary")

    gr.Examples(
        examples=example_urls,
        inputs=[url_label, url_input],
        label="サンプル画像のURL (クリックして入力)",
    )

    # --- 3. SVD結果を保持する State ---
    # gr.State を使用して SVD の結果 (U, S, Vt) を保持
    svd_state = gr.State(value={"U": None, "S": None, "Vt": None})
    # オリジナル画像も保持
    original_image_state = gr.State(value=None)

    # --- 4. 結果表示セクション ---
    with gr.Row():
        original_image_display = gr.Image(
            label="元のグレイスケール画像", type="pil", interactive=False
        )
        full_reconstructed_image_display = gr.Image(
            label="完全に再構成された画像 (全特異値使用)",
            type="pil",
            interactive=False,
        )

    # --- 5. 特異値選択スライダー ---
    singular_value_slider = gr.Slider(
        minimum=0,
        maximum=100,  # 初期値。画像読み込み後に最大値が更新される
        step=1,
        value=0,  # 初期値
        label="2. 再構成に使う特異値のインデックス k (0から始まる)",
        info="スライダーを動かして、再構成に使用する特異値を選択してください。",
    )

    # --- 6. 部分再構成の結果表示 ---
    gr.Markdown("### 選択したインデックス $k$ に基づく再構成")
    with gr.Row():
        recon_higher_image = gr.Image(
            label="上位 k+1 個の特異値で再構成", type="pil", interactive=False
        )
        recon_single_image = gr.Image(
            label="インデックス k の特異値のみで再構成",
            type="pil",
            interactive=False,
        )
        recon_lower_image = gr.Image(
            label="インデックス k+1 以降の特異値で再構成",
            type="pil",
            interactive=False,
        )

    with gr.Row():
        recon_higher_heatmap = gr.Plot(label="上位 k+1 個のヒートマップ")
        recon_single_heatmap = gr.Plot(label="インデックス k のヒートマップ")
        recon_lower_heatmap = gr.Plot(
            label="インデックス k+1 以降のヒートマップ"
        )

    # --- 7. 分解された行列の分析 ---
    gr.Markdown("### 分解された行列の分析")
    # 特異値プロット (線形、対数、累積寄与率)
    s_distribution_plot = gr.Plot(label="特異値 (Σ) の分布と累積寄与率")
    # U, S(タイル表示), Vt のヒートマップ
    u_heatmap_plot = gr.Plot(label="左特異ベクトル行列 U のヒートマップ")
    s_heatmap_plot = gr.Plot(label="特異値ベクトル S のヒートマップ (log10)")
    vt_heatmap_plot = gr.Plot(label="右特異ベクトル行列 Vt のヒートマップ")

    # --- 8. メモ・詳細情報表示 ---
    memo_display = gr.Markdown(label="メモ・詳細情報")

    # --- Gradio Event Handlers ---

    # --- イベント 1: URL入力・実行ボタン ---
    def handle_url_submit(image_url, current_k):
        """
        URLが入力されたときに呼び出される関数。
        画像を読み込み、SVDを実行し、初期表示を更新する。
        """
        print(f"画像URL '{image_url}' の処理を開始...")
        # 1. 画像読み込みとSVD実行
        original_img, svd_results, error_msg = load_image_and_perform_svd(
            image_url
        )

        if error_msg:
            print(f"エラー発生: {error_msg}")
            # エラーメッセージを表示し、既存の表示をクリア (またはエラー表示)
            gr.Warning(f"エラー: {error_msg}")  # Gradio 警告メッセージ
            # 空の要素またはプレースホルダを返す
            empty_img = Image.new("L", (100, 100), color="gray")
            empty_fig = plt.figure()
            return (
                empty_img,  # original_image_display
                empty_img,  # full_reconstructed_image_display
                0,
                1,
                0,  # singular_value_slider (min, max, value)
                empty_img,  # recon_higher_image
                empty_img,  # recon_single_image
                empty_img,  # recon_lower_image
                empty_fig,  # recon_higher_heatmap
                empty_fig,  # recon_single_heatmap
                empty_fig,  # recon_lower_heatmap
                empty_fig,  # s_distribution_plot
                empty_fig,  # u_heatmap_plot
                empty_fig,  # s_heatmap_plot
                empty_fig,  # vt_heatmap_plot
                f"処理中にエラーが発生しました:\n{error_msg}",  # memo_display
                {"U": None, "S": None, "Vt": None},  # svd_state (リセット)
                None,  # original_image_state (リセット)
            )

        print("SVD計算完了。")
        # 2. SVD結果から診断プロットを作成
        U_plot, S_plot, S_dist_plot, Vt_plot = create_diagnostic_plots(
            svd_results
        )
        print("診断プロット作成完了。")

        # 3. 全ての特異値を使って再構成
        full_reconstructed_img = reconstruct_image(
            svd_results["U"], svd_results["S"], svd_results["Vt"]
        )
        print("完全再構成画像作成完了。")

        # 4. スライダーの最大値を更新 (特異値の数 - 1)
        num_singular_values = len(svd_results["S"])
        slider_max = max(0, num_singular_values - 1)
        # 現在のスライダーの値が新しい最大値を超えないように調整
        adjusted_k = min(current_k, slider_max)
        slider_update = gr.Slider(
            maximum=slider_max, value=adjusted_k
        )  # Gradio 3.X+ update
        print(f"スライダー更新: max={slider_max}, value={adjusted_k}")

        # 5. 現在のスライダー値 (adjusted_k) で部分再構成を実行
        (img_h, img_s, img_l, plot_h, plot_s, plot_l, memo) = (
            create_reconstruction_details(svd_results, adjusted_k)
        )
        print(f"インデックス {adjusted_k} での部分再構成完了。")

        # 6. 結果を返す (Gradioコンポーネントに対応する順序で)
        return (
            original_img,  # original_image_display
            full_reconstructed_img,  # full_reconstructed_image_display
            slider_update,  # singular_value_slider (update object)
            img_h,  # recon_higher_image
            img_s,  # recon_single_image
            img_l,  # recon_lower_image
            plot_h,  # recon_higher_heatmap
            plot_s,  # recon_single_heatmap
            plot_l,  # recon_lower_heatmap
            S_dist_plot,  # s_distribution_plot
            U_plot,  # u_heatmap_plot
            S_plot,  # s_heatmap_plot
            Vt_plot,  # vt_heatmap_plot
            memo,  # memo_display
            svd_results,  # svd_state (更新)
            original_img,  # original_image_state (更新)
        )

    # --- イベント 2: スライダー変更 ---
    def handle_slider_change(k_index, current_svd_state):
        """
        スライダーの値が変更されたときに呼び出される関数。
        保持されているSVD結果を使って部分再構成を更新する。
        """
        print(f"スライダー変更: インデックス k = {k_index}")
        if current_svd_state is None or current_svd_state.get("S") is None:
            print("SVD結果が State に存在しません。")
            # エラーメッセージや空の結果を返す
            empty_img = Image.new("L", (50, 50), color="gray")
            empty_fig = plt.figure()
            return (
                empty_img,
                empty_img,
                empty_img,
                empty_fig,
                empty_fig,
                empty_fig,
                "画像を読み込んでからスライダーを操作してください。",
            )

        # SVD結果を使って部分再構成を計算
        (img_h, img_s, img_l, plot_h, plot_s, plot_l, memo) = (
            create_reconstruction_details(current_svd_state, k_index)
        )
        print(f"インデックス {k_index} での部分再構成更新完了。")

        # 部分再構成に関連するコンポーネントのみを更新
        return (
            img_h,  # recon_higher_image
            img_s,  # recon_single_image
            img_l,  # recon_lower_image
            plot_h,  # recon_higher_heatmap
            plot_s,  # recon_single_heatmap
            plot_l,  # recon_lower_heatmap
            memo,  # memo_display
        )

    # --- Component Mapping ---
    # URL入力・実行ボタン → handle_url_submit → 各表示コンポーネント + State更新
    submit_triggers = [url_input.submit, submit_button.click]
    outputs_for_submit = [
        original_image_display,
        full_reconstructed_image_display,
        singular_value_slider,  # スライダー自体を更新
        recon_higher_image,
        recon_single_image,
        recon_lower_image,
        recon_higher_heatmap,
        recon_single_heatmap,
        recon_lower_heatmap,
        s_distribution_plot,
        u_heatmap_plot,
        s_heatmap_plot,
        vt_heatmap_plot,
        memo_display,
        svd_state,
        original_image_state,  # Stateを更新
    ]
    gr.on(
        triggers=submit_triggers,
        fn=handle_url_submit,
        inputs=[url_input, singular_value_slider],  # 現在のスライダー値も渡す
        outputs=outputs_for_submit,
        # api_name="process_image" # APIとして公開する場合
    )

    # スライダー変更 → handle_slider_change → 部分再構成関連の表示更新
    outputs_for_slider = [
        recon_higher_image,
        recon_single_image,
        recon_lower_image,
        recon_higher_heatmap,
        recon_single_heatmap,
        recon_lower_heatmap,
        memo_display,
    ]
    singular_value_slider.change(
        fn=handle_slider_change,
        inputs=[
            singular_value_slider,
            svd_state,
        ],  # スライダーの値と SVD State を渡す
        outputs=outputs_for_slider,
        # api_name="update_reconstruction" # APIとして公開する場合
    )

# --- Application Launch ---
if __name__ == "__main__":
    # Gradioアプリケーションを起動
    demo.launch(
        share=False,  # Trueにするとパブリックリンクが生成される
        server_name="0.0.0.0",  # ローカルネットワークからアクセス可能にする場合
        # server_port=7860, # ポートを指定する場合
        # debug=True # デバッグモード
    )
