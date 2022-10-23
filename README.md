# GaussianDistillation

### Running CIFAR10 experiment

```python
# Train teacher
python utils/train_teacher.py CIFAR10 100

# Distill students
python distill_gaussian.py CIFAR10 200 200 --compare True --model_student resnet18
python distill_gaussian.py CIFAR10 200 200 --compare True --model_student mobilenetv2
```

### Citation
```
@inproceedings{
raikwar2022discovering,
title={Discovering and Overcoming Limitations of Noise-engineered Data-free Knowledge Distillation},
author={Piyush Raikwar and Deepak Mishra},
booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
year={2022},
url={https://openreview.net/forum?id=K8JngctQ2Tu}
}
```
