# TensorFlow implementation for CapsNet

## Test error rate on MNIST


|   	 Method        | Routing  | Reconstruction | Epochs | Batch Size | Test Error(%) | Observations                       |
|----------------------|----------|----------------|--------|------------|---------------|------------------------------------|
|   1Conv + 1PC + 1DC  |    2     |      yes       |   200  |     32     |      0.58     |   	            -                 |
|   2Conv + 1PC + 1DC  |    2     |      yes       |   200  |     32     |      0.57     |                  -	        	  | 
|   1Conv + 1PC + 1DC  |    3     |      yes       |    50  |     32     |      0.48     | bias for u_hat and 2-pixel shifted |