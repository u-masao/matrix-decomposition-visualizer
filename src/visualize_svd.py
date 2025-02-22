import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

def svd_image(file):
    # 画像を読み込む
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
    
    return gr.Image(value=reconstructed_img_pil, label="Reconstructed Image")

# GradioのUIを作成
with gr.Blocks() as demo:
    with gr.Row():
        image_input = gr.File(label="Drag and drop an image here or click to upload")
    
    # 画像がアップロードされたときの処理関数
    output_image = gr.Image(label="Reconstructed Image")
    
    # 画像アップロードとSVD分解結果の関連付け
    image_input.change(fn=svd_image, inputs=image_input, outputs=output_image)

# Gradioサーバーを起動
demo.launch(share=False)
