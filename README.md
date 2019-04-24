# TensorFlow implementation for CapsNet

## Test error rate on MNIST


|   	 Method        | Routing  | Reconstruction | Epochs | Batch Size | Test Error(%) |
|----------------------|----------|----------------|--------|------------|---------------|
|   1Conv + 1PC + 1DC  |    2     |      yes       |    40  |     32     |      0.64     |
|   2Conv + 1PC + 1DC  |    2     |      yes       |   200  |     32     |      0.57     | 