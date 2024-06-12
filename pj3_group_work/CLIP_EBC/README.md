* clip_based resnet-50, 用于人群计数

```python
from CLIP_EBC import get_model
...

# 创建模型实例，并将其移动到GPU上
device = "cuda:0"
current_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_dir, "CLIP_EBC/configs", f"reduction_8.json"), "r") as f:
    config = json.load(f)[str(4)]["shb"]
bins = config["bins"]["fine"]
anchor_points = config["anchor_points"]["fine"]["average"]
bins = [(float(b[0]), float(b[1])) for b in bins]
anchor_points = [float(p) for p in anchor_points]

model = get_model(backbone="clip_resnet50", input_size=448, reduction=8, bins=bins, anchor_points=anchor_points)
model = model.to(device)
...
```