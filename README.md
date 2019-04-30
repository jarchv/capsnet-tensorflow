# TensorFlow implementation for CapsNet

## Test error rate on MNIST


|   	 Method        | Routing  | Reconstruction | Epochs | Batch Size | Test Error(%) | Observations                         |
|----------------------|----------|----------------|--------|------------|---------------|--------------------------------------|
|   1Conv + 1PC + 1DC  |    2     |      yes       |   200  |     32     |      0.58     |   	            -                   |
|   2Conv + 1PC + 1DC  |    2     |      yes       |   200  |     32     |      0.57     |                  -	        	    | 
|   1Conv + 1PC + 1DC  |    3     |      yes       |    50  |     32     |      0.48     | Bias for u_hat and 2-pixel shifted   |
|   1Conv + 1PC + 1DC  |    3     |      yes       |    50  |     64     |      0.36     | +Exponential decay for learning rate |
|   1Conv + 1PC + 1DC  |    3     |      yes       |    50  |     64     |      0.35     | +Learning rate = 0.00005             |