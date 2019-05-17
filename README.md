# TensorFlow implementation of CapsNet

A Tensorflow implementation of CapsNet based on the paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829).

## Test error rate on MNIST

Test error on MNIST reported by 3 trials. The average and standard desviation results are reported by 3 trials.

| Method  | Routing | Reconstruction | Epochs | Batch Size | Test Error(%) |     Paper)  |
|---------|---------|----------------|--------|------------|---------------|-------------|
| CapsNet |    3    |      No        |   50   |     125    | 0.343(0.012)  | 0.25(0.036) |