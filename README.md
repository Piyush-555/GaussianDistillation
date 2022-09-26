# GaussianDistillation

### Code for CIFAR10 experiment

**Requirement:** numpy pytorch torchvision matplotlib fire

```python
python utils/train_teacher.py CIFAR10 100
python distill_gaussian.py CIFAR10 200 200 --compare True --model_student resnet18
python distill_gaussian.py CIFAR10 200 200 --compare True --model_student mobilenetv2
```
