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
@inproceedings{NEURIPS2022_1f96b24d,
 author = {Raikwar, Piyush and Mishra, Deepak},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {4902--4912},
 publisher = {Curran Associates, Inc.},
 title = {Discovering and Overcoming Limitations of Noise-engineered Data-free Knowledge Distillation},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/1f96b24df4b06f5d68389845a9a13ed9-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
}
```
