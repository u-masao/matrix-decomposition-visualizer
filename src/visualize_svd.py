from io import BytesIO

import gradio as gr
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.cm import ScalarMappable
from PIL import Image

# init U, Sigma, Vt
U = None
S = None
Vt = None

# 例として提供された画像URLのリスト
example_urls = [
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSoki9ZKPtsoYw1z_A2"
    "DZQIpMnObustPL7OKA&s",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRleTnZSi7n7K28PycQ"
    "4IduBkFTn3qyd5t0QA&s",
    "https://images.rawpixel.com/image_800/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS"
    "8yMDIyLTA1L2lzODk3OS1pbWFnZS1rd3Z5ZW5sZi5qcGc.jpg",
    "https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYW"
    "dlcy93ZWJzaXRlX2NvbnRlbnQvcHgxMjU3NDI0LWltYWdlLWpvYjYzMC1uLWwwZzA4Zzc2Lm"
    "pwZw.jpg",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR5dXqHnnrYPqQNSrEW"
    "j6_TH3rLEF1FcCCatsAm6pcx_crAiNmWl2pdfTXNNe04_JZMnio&usqp=CAU",
    "https://live.staticflickr.com/7497/15364254203_e6d7a2465b_b.jpg",
]


def plot_heatmap(data):
    """
    plot heatmap and histogram
    """
    plt.close()

    figsize = (12, 4)
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # カラーマップを設定
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=data.min(), vmax=data.max())
    sm = ScalarMappable(cmap=cmap, norm=norm)

    # ヒートマップを表示
    ax[0].imshow(data, cmap=cmap, norm=norm)
    sns.histplot(data.flatten(), kde=True, ax=ax[1], bins=30)
    fig.colorbar(sm, ax=ax[0])
    fig.tight_layout()

    return fig


def plot_diag(data):
    """
    plot diag
    """
    plt.close()

    figsize = (12, 8)
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax[0].plot(data)
    ax[0].set_ylabel("特異値")
    ax[1].plot(data)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("特異値(対数)")
    ax[1].set_xlabel("インデックス")
    ax[0].grid()
    ax[1].grid()

    return fig


def reconstruct(U, S, Vt, singular_index: int):
    """
    reconstruct image from U, S, Vt

    """

    # vector to matrix about Sigma
    S_diag = np.zeros((U.shape[0], Vt.shape[1]))

    # 対角成分を埋める
    np.fill_diagonal(S_diag, S)

    # UとS_diagとVtの積で元の画像を再構成
    reconstructed_img = (255.0 * U @ S_diag @ Vt).clip(0, 255).astype("uint8")

    # PIL形式に戻す
    reconstructed_img_pil = Image.fromarray(reconstructed_img)

    return reconstructed_img_pil


def update_image(image_url, singular_index):
    """
    update_image gradio components callback
    """
    comps = load_and_decompose_image(image_url)
    comps += decompose_singular_index(singular_index)
    return comps


def load_and_decompose_image(image_url):
    """
    load image and decompose
    """

    global U, S, Vt
    # 画像データを取得してImageオブジェクトに変換
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("L")
    img_array = np.array(img) / 255.0

    # SVD分解を実行
    U, S, Vt = np.linalg.svd(img_array)

    # reconstruct
    reconstructed_img = reconstruct(U, S, Vt, singular_index)

    # Uのビットマップ画像を作成
    U_fig = plot_heatmap(U)

    # log10(S)のビットマップ画像を作成
    S_bitmap = np.log10(S)

    # 横幅を広げる
    S_bitmap = np.tile(S_bitmap[:, np.newaxis], min(img.size))
    S_fig = plot_heatmap(S_bitmap)
    S_diag = plot_diag(S)

    # Vのビットマップ画像を作成
    V_fig = plot_heatmap(Vt)

    return (
        gr.Image(value=img, label="グレイスケール画像（オリジナル）"),
        gr.Image(value=reconstructed_img, label="再構成画像"),
        gr.Plot(value=U_fig, label="左特異ベクトルの行列 U のヒートマップ"),
        gr.Plot(value=S_fig, label="特異行列 Sigma の対角成分のヒートマップ"),
        gr.Plot(value=S_diag, label="特異行列 Sigma の対角成分のプロット"),
        gr.Plot(value=V_fig, label="右特異ベクトルの行列 Vt のヒートマップ"),
    )


def decompose_singular_index(singular_index):
    """
    decompose singular_index

    """
    global U, S, Vt

    # input validation
    for x in [U, S, Vt]:
        if x is None:
            return tuple([None] * 3)

    reconstructed_img_positive = reconstruct(
        U,
        np.where(np.arange(len(S)) < singular_index, S, 0),
        Vt,
        singular_index,
    )

    reconstructed_img_single = reconstruct(
        U,
        np.where(np.arange(len(S)) == singular_index, S, 0),
        Vt,
        singular_index,
    )

    reconstructed_img_negative = reconstruct(
        U,
        np.where(np.arange(len(S)) > singular_index, S, 0),
        Vt,
        singular_index,
    )

    # make memo
    memo = ""
    memo += f"{pd.Series(S).describe().to_dict()=}\n\n"
    memo += f"{U.shape=}\n\n"
    memo += f"{S.shape=}\n\n"
    memo += f"{Vt.shape=}\n\n"
    memo += f"{U[:,singular_index]=}\n\n"  # noqa: E231
    memo += f"{S[singular_index]=}\n\n"
    memo += f"{Vt[singular_index,:]=}\n\n"  # noqa: E231

    return (
        gr.Image(
            value=reconstructed_img_positive, label="大きい特異値の再構成画像"
        ),
        gr.Image(
            value=reconstructed_img_single, label="指定の特異値の再構成画像"
        ),
        gr.Image(
            value=reconstructed_img_negative, label="小さい特異値の再構成画像"
        ),
        gr.Plot(
            value=plot_heatmap(np.array(reconstructed_img_positive)),
            label="大きい特異値のヒートマップ",
        ),
        gr.Plot(
            value=plot_heatmap(np.array(reconstructed_img_single)),
            label="指定の特異値のヒートマップ",
        ),
        gr.Plot(
            value=plot_heatmap(np.array(reconstructed_img_negative)),
            label="小さい特異値のヒートマップ",
        ),
        gr.Markdown(memo, label="memo"),
    )


# GradioのUIを作成

wikipedia = (
    "https://ja.wikipedia.org/wiki/"
    "%E7%89%B9%E7%95%B0%E5%80%A4%E5%88%86%E8%A7%A3"
)

with gr.Blocks(fill_width=True) as demo:
    gr.Markdown(
        f"""
    # 特異値分解の可視化ツール➗

    このツールは行列を分解する手法の一つである[特異値分解]({wikipedia})
    (singular value decomposition, SVD) を直感的に理解するためのものです。
    """
    )

    with gr.Accordion(label="使い方", open=False):
        gr.Markdown(
            """
            指定のURLの画像をグレイスケールに変換して行列とみなします。
            グレイスケールの数値は、0 から 255 の値を持ちますが、
            行列分解処理時には 0.0 から 1.0 に正規化しています。
            特異値は大きいものから降順にソートされています。

            1. 画像 URL の入力

            「サンプル画像のURL」のどれかをクリックするか、
            「1. 画像の URL」 入力欄に画像のURLを入力します。

            2. 画像が読み込めたことを確認し、再構成に成功していることを味わいます


            3. 利用する特異値の変更

            「再構成に利用する特異値のインデックス」スライダーを動かして、
            再構成画像の変化を味わいます。

            - 大きい特異値の再構成画像は、指定のインデックスよりも大きい値の特異値
            を利用した再構成画像です。

            - 指定の特異値の再構成画像は、指定のインデックスの特異値と特異ベクトル
            を使った再構成画像です。

            - 小さい特異値の再構成画像は、指定のインデックスよりも小さい値の特異値
            を利用した再構成画像です。


            """
        )
    with gr.Row():
        url_input = gr.Textbox(
            lines=1,
            placeholder="Enter image URL here...",
            scale=5,
            label="1. 画像の URL",
        )
        submit_button = gr.Button(value="submit", scale=1)

    with gr.Row():
        gr.Examples(
            examples=example_urls,
            inputs=url_input,
            label="サンプル画像のURL",
        )

    with gr.Row():
        singular_index = gr.Slider(
            minimum=0,
            maximum=100,
            step=1,
            label="2. 再構成に利用する特異値のインデックス",
        )

    with gr.Row():
        positive_image, single_image, negative_image = (
            gr.Image(label="大きい特異値の再構成画像"),
            gr.Image(label="指定の特異値の再構成画像"),
            gr.Image(label="小さい特異値の再構成画像"),
        )

    with gr.Row():
        positive_heatmap, single_heatmap, negative_heatmap = (
            gr.Plot(label="大きい特異値のヒートマップ"),
            gr.Plot(label="指定の特異値のヒートマップ"),
            gr.Plot(label="小さい特異値のヒートマップ"),
        )
    with gr.Row():
        original_image, output_image = (
            gr.Image(label="グレイスケール画像（オリジナル）"),
            gr.Image(label="再構成画像"),
        )

    gr.Markdown("## 分解された行列の分析")

    S_hist, U_bitmap, S_bitmap, V_bitmap = (
        gr.Plot(label="特異行列 Sigma の対角成分のプロット"),
        gr.Plot(label="左特異ベクトルの行列 U のヒートマップ"),
        gr.Plot(label="特異行列 Sigma の対角成分(特異値)のヒートマップ"),
        gr.Plot(label="右特異ベクトルの行列 Vt のヒートマップ"),
    )

    with gr.Row():
        memo = gr.Markdown("")

    # define outputs
    outputs_load = [
        original_image,
        output_image,
        U_bitmap,
        S_bitmap,
        S_hist,
        V_bitmap,
    ]

    outputs_decompose = [
        positive_image,
        single_image,
        negative_image,
        positive_heatmap,
        single_heatmap,
        negative_heatmap,
        memo,
    ]

    # load image and decompose
    gr.on(
        [
            url_input.submit,
            url_input.change,
            submit_button.click,
        ],
        fn=update_image,
        inputs=[url_input, singular_index],
        outputs=outputs_load + outputs_decompose,
    )

    # URLから画像を取得してSVD分解する関数のトリガー設定
    gr.on(
        [
            singular_index.change,
        ],
        fn=decompose_singular_index,
        inputs=[singular_index],
        outputs=outputs_decompose,
    )

if __name__ == "__main__":
    # Gradioサーバーを起動
    demo.launch(
        share=False,
        server_name="0.0.0.0",
    )
