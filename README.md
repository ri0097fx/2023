# 2023
Download ImageNet-class JSON file:
```
!wget https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
```

Read JSON file:
```
import json
class_index = json.load(open('imagenet_class_index.json', 'r'))
print(class_index)
```

