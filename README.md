# TensorFlow implementation of CapsNet

A Tensorflow implementation of CapsNet based on the paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829).

## Test error rate on MNIST


| Method  | Routing | Reconstruction | Epochs | Batch Size | Test Error(%) |     Paper   |
|---------|---------|----------------|--------|------------|---------------|-------------|
| CapsNet |    3    |      yes       |   50   |     100    | 0.40(0.00)    | 0.25(0.005) |