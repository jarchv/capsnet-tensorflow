# TensorFlow implementation of CapsNet

A Tensorflow implementation of CapsNet based on the paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829).

## Test error rate on MNIST


| Method  | Routing | Reconstruction | Epochs | Batch Size | Test Error(%) |
|---------|---------|----------------|--------|------------|---------------|
| CapsNet |    3    |      yes       |   50   |     64     |      0.43     |