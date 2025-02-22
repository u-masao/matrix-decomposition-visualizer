# UVパッケージ管理を使用した環境設定
import uv  # UVモジュールのインポート

class Visualizer:
    def __init__(self):
        self.config = uv.config.get()
