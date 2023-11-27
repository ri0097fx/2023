# 2023
## Week 1
### 画像認識
GitHubnのレポジトリをクローンする
```python
!git clone https://github.com/ri0097fx/2023.git
```
ディレクトリを移動する
```python
cd 2023
```
ImageNetのクラス情報(JSON)をダウンロードする
```python
!wget https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
```
JSONファイルを読み込む
```python
import json
class_index = json.load(open('imagenet_class_index.json', 'r'))
print(class_index)
```
辞書のキーをstring型からint型へ変換する
```python
labels = {int(key):value for (key, value) in class_index.items()}
print(labels[332])
```
ImageNetで学習済みの画像認識モデルを読み込む
```python
from torchvision import models
from utils import *

model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).eval()
```
画像を読み込み、テンソルデータに変換する
```python
img_path = './sample_data/sample.jpg'
tensor_img = make_tensor_img(img_path)
```
モデルに入力し、出力を表示する
```python
out = model(tensor_img)
predict = out.argmax(-1)
for i in predict:
    print(labels[i.item()][-1])
```