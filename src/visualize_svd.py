from io import BytesIO

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
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

    # data_rows = data.shape[0]
    # data_cols = data.shape[1]
    # figsize = (data_cols / 40 + 3.0, data_rows / 40)
    figsize = (12, 4)
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # カラーマップを設定
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=data.min(), vmax=data.max())
    sm = ScalarMappable(cmap=cmap, norm=norm)

    # ヒートマップを表示
    ax[0].imshow(data, cmap=cmap, norm=norm)
    ax[1].hist(data.flatten(), bins=51)
    fig.colorbar(sm, ax=ax[0])
    fig.tight_layout()

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

    # Vのビットマップ画像を作成
    V_fig = plot_heatmap(Vt)

    # make memo
    memo = ""
    memo += f"{pd.Series(S).describe().to_dict()=}\n\n"
    memo += f"{U.shape=}\n\n"
    memo += f"{S.shape=}\n\n"
    memo += f"{Vt.shape=}\n\n"

    return (
        gr.Image(value=img, label="Grayscale Image"),
        gr.Image(value=reconstructed_img, label="Reconstructed Image"),
        gr.Plot(value=U_fig, label="U Bitmap"),
        gr.Plot(value=S_fig, label="S bitmap"),
        gr.Plot(value=V_fig, label="V Bitmap"),
        gr.Markdown(memo, label="memo"),
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

    return (
        gr.Image(value=reconstructed_img_positive, label="Positive Image"),
        gr.Image(value=reconstructed_img_single, label="Single Image"),
        gr.Image(value=reconstructed_img_negative, label="Negative Image"),
    )


# GradioのUIを作成
with gr.Blocks() as demo:
    with gr.Row():
        url_input = gr.Textbox(
            lines=1, placeholder="Enter image URL here...", scale=5
        )
        submit_button = gr.Button(value="submit", scale=1)
    with gr.Row():
        gr.Examples(examples=example_urls, inputs=url_input)
    with gr.Row():
        singular_index = gr.Slider(
            minimum=0,
            maximum=100,
            step=1,
            label="再構成に利用する特異値のインデックス",
        )
    with gr.Row():
        positive_image, single_image, negative_image = (
            gr.Image(label="Positive Image"),
            gr.Image(label="Single Image"),
            gr.Image(label="Nagative Image"),
        )
    with gr.Row():
        original_image, output_image = (
            gr.Image(label="Gray scale Image"),
            gr.Image(label="Reconstructed Image"),
        )
    gr.Markdown("## Analysis U, SIGMA, Vt ")
    U_bitmap, S_bitmap, V_bitmap = (
        gr.Plot(label="U Bitmap"),
        gr.Plot(label="S Bitmap"),
        gr.Plot(label="V Bitmap"),
    )
    with gr.Row():
        memo = gr.Markdown("")

    # load image and decompose
    for func in [
        url_input.submit,
        url_input.change,
        submit_button.click,
    ]:
        func(
            update_image,
            inputs=[url_input, singular_index],
            outputs=[
                original_image,
                output_image,
                U_bitmap,
                S_bitmap,
                V_bitmap,
                memo,
                positive_image,
                single_image,
                negative_image,
            ],
        )

    # URLから画像を取得してSVD分解する関数のトリガー設定
    for func in [
        singular_index.change,
    ]:
        func(
            decompose_singular_index,
            inputs=[singular_index],
            outputs=[
                positive_image,
                single_image,
                negative_image,
            ],
        )

if __name__ == "__main__":
    # Gradioサーバーを起動
    demo.launch(
        share=False,
        server_name="0.0.0.0",
    )
