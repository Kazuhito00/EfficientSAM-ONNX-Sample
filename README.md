# EfficientSAM-ONNX-Sample
[yformer/EfficientSAM](https://github.com/yformer/EfficientSAM) をColaboraotry上でONNX推論するサンプルです。<br>

https://github.com/Kazuhito00/EfficientSAM-ONNX-Sample/assets/37477845/5dd6f004-2be2-4b39-a4d2-2b8f2beb576b

# Usage
#### ONNX変換、推論テスト
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhito00/EfficientSAM-ONNX-Sample/blob/main/EfficientSAM-ONNX-Colaboratory-Sample.ipynb)<br>
Colaboratoryでノートブックを開き、上から順に実行してください。<br>

#### 簡易デモ
以下コマンドでデモを起動してください。<br>
左クリックでプロンプト座標を追加、右クリックでプロンプト座標を削除します。<br>
また、キーボード1～3で、クリック時に追加する座標のタイプを変更します（1：対象座標、2：非対象座標、3：対象バウンディングボックス）
```
python demo.py
```
* --image<br>
画像ファイルの指定<br>
デフォルト：sample.png
* --encoder<br>
エンコーダーONNXファイルのパス<br>
デフォルト：onnx_model/efficient_sam_vitt_encoder.onnx
* --decoder<br>
デコーダーONNXファイルのパス<br>
デフォルト：onnx_model/efficient_sam_vitt_decoder.onnx

# Note
サンプルの画像は[ぱくたそ](https://www.pakutaso.com/)様の「[トゲトゲのサボテンとハリネズミ](https://www.pakutaso.com/20190257050post-19488.html)」を使用しています。

# License 
EfficientSAM-ONNX-Sample is under [Apache-2.0 license](LICENSE).

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
