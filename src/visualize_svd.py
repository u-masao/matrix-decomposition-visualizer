import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

def svd_image(file):
    # 画像をURLから取得する
    img = Image.open(file).convert('L')
    img = np.array(img)
    
    # SVD分解
    U, S, Vt = np.linalg.svd(img)
    
    # 特異値の最大値を取得
    max_singular_value = np.max(S)
    
    # 特異値行列を再構成
    S_diag = np.zeros((U.shape[0], Vt.shape[1]))
    np.fill_diagonal(S_diag, S)
    
    # UとS_diagとVtの積で元の画像を再構成
    reconstructed_img = (U @ S_diag @ Vt).clip(0, 255).astype('uint8')
    
    # 結果を表示するために画像をPIL形式に戻す
    reconstructed_img_pil = Image.fromarray(reconstructed_img)
    
    # Uのビットマップ画像を作成
    U_bitmap = (U * 127 + 128).clip(0, 255).astype('uint8')
    U_bitmap_pil = Image.fromarray(U_bitmap)
    
    # Vのビットマップ画像を作成
    V_bitmap = (Vt * 127 + 128).clip(0, 255).astype('uint8')
    V_bitmap_pil = Image.fromarray(V_bitmap)
    
    # Sのビットマップ画像を作成
    S_bitmap = (S / max_singular_value * 255).clip(0, 255).astype('uint8')
    S_bitmap_pil = Image.fromarray(S_bitmap)
    
    return gr.Image(value=reconstructed_img_pil, label="Reconstructed Image"), \
           gr.Image(value=U_bitmap_pil, label="U Bitmap"), \
           gr.Image(value=V_bitmap_pil, label="V Bitmap"), \
           gr.Image(value=S_bitmap_pil, label="S Bitmap")

# GradioのUIを作成
with gr.Blocks() as demo:
    with gr.Row():
        image_input = gr.File(label="Drag and drop an image here or click to upload")
    
    # 画像がアップロードされたときの処理関数
    output_image, U_bitmap, V_bitmap, S_bitmap = gr.Image(label="Reconstructed Image"), \
                                                 gr.Image(label="U Bitmap"), \
                                                 gr.Image(label="V Bitmap"), \
                                                 gr.Image(label="S Bitmap")
    
    # 画像アップロードとSVD分解結果の関連付け
    image_input.change(fn=svd_image, inputs=image_input, outputs=[output_image, U_bitmap, V_bitmap, S_bitmap])

# Gradioサーバーを起動
demo.launch(share=False)
