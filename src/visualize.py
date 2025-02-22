import gradio as gr

# Gradio のブロックスタイルの UI を作成
with gr.Blocks() as demo:
    # 画像を受け付けるためのドラッグアンドドロップ領域を作成
    with gr.Row():
        image_input = gr.File(label="Drag and drop an image here or click to upload")
    
    # 画像がアップロードされたときの処理関数
    def process_image(file):
        return file

    # 処理結果の表示
    output_image = gr.Image(label="Processed Image")

    # 画像アップロードと処理結果の関連付け
    image_input.change(fn=process_image, inputs=image_input, outputs=output_image)

# Gradio サーバーを起動
demo.launch(share=False)
