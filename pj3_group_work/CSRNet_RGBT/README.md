* 4 channels csrnet, 用于人群计数

```python
from CSRNet_RGBT.csrnet_rgbt import CSRNet
...

# 创建模型实例，并将其移动到GPU上
model = CSRNet()
...
```

lr start : 1e-5