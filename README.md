# TensorFlow implementation of CapsNet

A Tensorflow implementation of CapsNet based on the paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829).

## Test error rate on MNIST

Test error on MNIST reported by 3 trials. The average and standard desviation results are reported by 3 trials.

| Method  | Routing | Reconstruction | Epochs | Batch Size | Params | Test Error(%) |     Paper     |
| CapsNet |    3    |      No        |   50   |     125    |**6.8M**| 0.343(0.012)  |  0.35(0.036)  |
| CapsNet |    3    |      Yes       |   --   |     125    |**8.2M**|   ---------   |**0.25(0.005)**| 

### Notes

* The number parameters are same for both setups of the paper.  
