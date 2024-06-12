* 纯resnet-50，用于人群计数

```python
from Res50.model.Res50 import Res50
...

# 创建模型实例，并将其移动到GPU上
model = Res50()
```

lr_start: 1e-3