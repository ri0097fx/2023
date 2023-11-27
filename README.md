# 2023
## Week 1
### 画像認識
レポジトリをクローンする
```python
!git clone https://github.com/ri0097fx/2023.git
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
