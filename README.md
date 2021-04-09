# 逮虾户

## 模型说明

最终排名：137/4285

- backnone: efficientb6
- model: unet++
- 0.8/0.2 训练/验证数据
- dice-loss + Focal-loss 联合Loss
- optimize: adamW
- warmUpConsineScheduler

超参数详见./utils/option.py
epoches 100

## 代码运行说明
### 环境:
- torch>1.6
- segmentation_models_pytorch
- pytorch_toolbelt
