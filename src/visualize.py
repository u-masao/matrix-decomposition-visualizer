<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>Gradioで文字を表示</title>
    <!-- Gradioライブラリの読み込み -->
    <script src="https://cdn.jsdelivr.net/npm/@gradio/core"></script>
</head>
<body>
    <div id="output"></div>
    <script type="module">
        // Gradioを使用してテキストを表示
        import { createElement } from 'https://cdn.jsdelivr.net/npm/@gradio/core';

        const output = document.getElementById('output');
        const textElement = createElement('p', { text: 'こんにちは、世界！' });
        output.appendChild(textElement);
    </script>
</body>
</html>
