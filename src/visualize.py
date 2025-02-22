import gradio as gr

def display_text(text):
    return text

iface = gr.Interface(fn=display_text, inputs="text", outputs="text")
iface.launch()
