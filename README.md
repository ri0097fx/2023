# 2023
##基礎実習１
Download ImageNet-class JSON file:
```python
!wget https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
```

Load JSON file:
```python
import json
class_index = json.load(open('imagenet_class_index.json', 'r'))
print(class_index)
```

Convert dict-keys str to int:
```python
labels = {int(key):value for (key, value) in class_index.items()}
print(labels[332])
```
