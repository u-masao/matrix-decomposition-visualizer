import gradio as gr
import numpy as np
from PIL import Image
import requests
from io import BytesIO

def svd_image(image_url):
    # 画像データを取得して Image オブジェクトにすること
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("L")
    img = np.array(img)

    # SVD分解
    U, S, Vt = np.linalg.svd(img)

    # 特異値の最大値を取得
    max_singular_value = np.max(S)

    # 特異値行列を再構成
    S_diag = np.zeros((U.shape[0], Vt.shape[1]))
    np.fill_diagonal(S_diag, S)

    # UとS_diagとVtの積で元の画像を再構成
    reconstructed_img = (U @ S_diag @ Vt).clip(0, 255).astype("uint8")

    # 結果を表示するために画像をPIL形式に戻す
    reconstructed_img_pil = Image.fromarray(reconstructed_img)

    # Uのビットマップ画像を作成
    U_bitmap = (U * 127 + 128).clip(0, 255).astype("uint8")
    U_bitmap_pil = Image.fromarray(U_bitmap)

    # Vのビットマップ画像を作成
    V_bitmap = (Vt * 127 + 128).clip(0, 255).astype("uint8")
    V_bitmap_pil = Image.fromarray(V_bitmap)

    # Sのビットマップ画像を作成
    S_bitmap = (S / max_singular_value * 255).clip(0, 255).astype("uint8")
    S_bitmap_pil = Image.fromarray(S_bitmap)

    return (
        gr.Image(value=reconstructed_img_pil, label="Reconstructed Image"),
        gr.Image(value=U_bitmap_pil, label="U Bitmap"),
        gr.Image(value=S_bitmap_pil, label="S Bitmap"),
        gr.Image(value=V_bitmap_pil, label="V Bitmap"),
    )

# GradioのUIを作成
with gr.Blocks() as demo:
    with gr.Row():
        url_input = gr.Textbox(lines=1, placeholder="Enter image URL here...")

    # 画像がアップロードされたときの処理関数
    output_image, U_bitmap, S_bitmap, V_bitmap = (
        gr.Image(label="Reconstructed Image"),
        gr.Image(label="U Bitmap"),
        gr.Image(label="S Bitmap"),
        gr.Image(label="V Bitmap"),
    )

    # URLから画像を取得してSVD分解する関数
    url_input.submit(
        svd_image,
        inputs=url_input,
        outputs=[output_image, U_bitmap, S_bitmap, V_bitmap],
    )


# Gradioサーバーを起動
demo.launch(share=False)
