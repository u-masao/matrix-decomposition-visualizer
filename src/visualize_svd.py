import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

def svd_image(file):
    # 画像を読み込む
    img = Image.open(file).convert('L')
    img_array = np.array(img)
    
    # SVD分解
    U, S, Vt = np.linalg.svd(img_array)
    
    # 再構成画像 (k=50までの特異値を使用)
    k = 50
    img_reconstructed = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
    
    # 元画像と再構成画像を表示
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(img_reconstructed, cmap='gray')
    axs[1].set_title('Reconstructed Image (k=50)')
    
    # 画像を表示
    plt.show()
    
    return fig

# Gradio UIの作成
with gr.Blocks() as demo:
    with gr.Row():
        image_input = gr.File(label="Drag and drop an image here or click to upload")
    
    def process_image(file):
        return svd_image(file)

    output_fig = gr.Plot(label="SVD Decomposed Image")
    image_input.change(fn=process_image, inputs=image_input, outputs=output_fig)

# Gradioサーバーの起動
demo.launch(share=False)
