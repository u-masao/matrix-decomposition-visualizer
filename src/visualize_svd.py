from io import BytesIO

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.cm import ScalarMappable
from PIL import Image

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
    data_rows = data.shape[0]
    data_cols = data.shape[1]

    fig, ax = plt.subplots(figsize=(data_cols / 40 + 3.0, data_rows / 40))

    # Create a ScalarMappable to get the colorbar
    cmap = plt.get_cmap("viridis")  # You can choose any colormap here
    norm = plt.Normalize(vmin=data.min(), vmax=data.max())
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # This is needed for the colorbar to work

    # Display the heatmap
    ax.imshow(data, cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax)  # , fraction=0.046, pad=0.035)
    fig.tight_layout()

    return fig


def svd_image(image_url, ignore_singular):
    # 画像データを取得して Image オブジェクトにすること
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("L")
    img_array = np.array(img) / 255.0

    # SVD分解
    U, S, Vt = np.linalg.svd(img_array)

    # 特異値行列を再構成
    S_diag = np.zeros((U.shape[0], Vt.shape[1]))
    S_subset = np.where(np.arange(len(S)) == ignore_singular, S, 0)
    np.fill_diagonal(S_diag, S_subset)

    # UとS_diagとVtの積で元の画像を再構成
    reconstructed_img = (255.0 * U @ S_diag @ Vt).clip(0, 255).astype("uint8")

    # 結果を表示するために画像をPIL形式に戻す
    reconstructed_img_pil = Image.fromarray(reconstructed_img)

    # Uのビットマップ画像を作成
    U_bitmap = (U * 127 + 128).clip(0, 255).astype("uint8")
    U_fig = plot_heatmap(U_bitmap)

    # Sのビットマップ画像を作成
    max_singular_value = np.log10(np.max(S))
    S_bitmap = (
        (np.log10(S) / max_singular_value * 255).clip(0, 255).astype("uint8")
    )
    S_bitmap = np.log10(S)
    S_bitmap = np.tile(S_bitmap[:, np.newaxis], 10)
    S_fig = plot_heatmap(S_bitmap)

    # Vのビットマップ画像を作成
    V_bitmap = (Vt * 127 + 128).clip(0, 255).astype("uint8")
    V_fig = plot_heatmap(V_bitmap)

    memo = ""
    memo += f"{pd.Series(S).describe().to_dict()=}\n"

    return (
        gr.Image(value=img, label="Original Image"),
        gr.Image(value=reconstructed_img_pil, label="Reconstructed Image"),
        gr.Plot(value=U_fig, label="U Bitmap"),
        gr.Plot(S_fig, label="S bitmap"),
        gr.Plot(value=V_fig, label="V Bitmap"),
        gr.Markdown(memo, label="memo"),
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
        ignore_singular = gr.Slider(
            minimum=0,
            maximum=100,
            step=1,
            label="再構成に利用する特異値のインデックス",
        )
    with gr.Row():
        memo = gr.Markdown("")
    with gr.Row():
        original_image, output_image = (
            gr.Image(label="Original Image"),
            gr.Image(label="Reconstructed Image"),
        )
    with gr.Row():
        U_bitmap, S_bitmap, V_bitmap = (
            gr.Plot(label="U Bitmap"),
            gr.Plot(label="S Bitmap"),
            gr.Plot(label="V Bitmap"),
        )

    # URLから画像を取得してSVD分解する関数

    for func in [
        url_input.submit,
        url_input.change,
        submit_button.click,
        ignore_singular.change,
    ]:
        func(
            svd_image,
            inputs=[url_input, ignore_singular],
            outputs=[
                original_image,
                output_image,
                U_bitmap,
                S_bitmap,
                V_bitmap,
                memo,
            ],
        )

if __name__ == "__main__":
    # Gradioサーバーを起動
    demo.launch(share=False)
