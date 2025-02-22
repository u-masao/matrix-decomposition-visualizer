import gradio as gr

# Gradio のブロックスタイルの UI を作成
with gr.Blocks() as demo:
    # 2つの数値入力フィールドを作成
    num1 = gr.Number(label="Input 1")
    num2 = gr.Number(label="Input 2")

    # 和を計算する関数
    def add_numbers(n1, n2):
        return n1 + n2

    # 計算結果の表示テキストボックスを作成
    result = gr.Textbox(label="Result")

    # 数値入力と計算結果の関連付け
    num1.change(fn=add_numbers, inputs=[num1, num2], outputs=result)

# Gradio サーバーを起動
demo.launch(share=False)
