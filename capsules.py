import tensorflow as tf

class CapsNet:
  def __init__(	self, mode = 'conv'):
	  self.mode = mode
    
  def layer(self,
            inputs      = None, 
            filters     = None,
            kernel_size = 9,
            strides     = (1, 1),
            padding     = 'valid',
            activation  = None,
            use_bias    = True,
            trainable 	= True,
            caps_units 	= None, 
            caps_dim    = None,
            rounds      = 3,
            batch_size  = None,
            name        = None):

    if self.mode == 'conv2d':
      output = tf.layers.conv2d(inputs      = inputs,
                                filters     = filters,
                                kernel_size = kernel_size,
                                strides     = strides,
                                padding     = padding,
                                activation  = activation,
                                use_bias    = use_bias,
                                trainable   = trainable,
                                name        = name
                                )

    elif self.mode == 'primary':
      conv  =  tf.layers.conv2d(inputs      = inputs,
                                filters     = caps_units * caps_dim,
                                kernel_size = kernel_size,
                                strides     = strides,
                                padding     = padding,
                                activation  = activation,
                                use_bias    = use_bias,
                                trainable   = trainable,
                                name        = name)

      grid_dim  = conv.shape[-2].value
      assert (grid_dim == conv.shape[-3].value)

      s = tf.reshape(conv, [-1, grid_dim * grid_dim * caps_units, caps_dim])
      output = self.squash(s, name = 'PrimaryCapsule_output')

    elif self.mode == 'digit':
      # inputs.shape = [?, 1152, 8]
      
      self.input_caps_units = inputs.shape[1].value
      self.input_caps_dim   = inputs.shape[2].value

      with tf.name_scope(name):
        inputs_tile     = tf.expand_dims(inputs,  2, name = 'inputs_tile') # [?, 1152, 1, 8]

        self.caps_units = caps_units
        self.caps_dim   = caps_dim
        self.rounds     = rounds

        with tf.variable_scope('routing'):
          self.batch_size = batch_size
					
          # "Initial logits b_ij are the log prior probabilities that capsule i shoould be coupled to capsule j"
          # b_ij => [#Capsules_i, #Capsules_j] :[1152, 10]
				

          output = self.routing(inputs_tile, inputs.shape.as_list())

    return output		

  def squash(self, s, axis = -1, epsilon = 1e-7, name = 'squash'):
    with tf.name_scope(name):
      s_squared_norm  = tf.reduce_sum(tf.square(s), axis = axis, keepdims = True)
      squash_factor   = s_squared_norm / (1.0 + s_squared_norm)
      squash_unit_vec = s / tf.sqrt(s_squared_norm + epsilon)
      
      return  squash_factor * squash_unit_vec

  def routing(self, inputs_tile, inputs_shape):
    W_init = tf.random_normal( 
              shape  = (1, self.input_caps_units, self.caps_dim * self.caps_units, self.input_caps_dim), #[1, 1152, 160, 8]
              stddev = 0.01, 
              dtype  = tf.float32,
              name   = 'W_init')
		
    W = tf.Variable(W_init, name = 'W') #[1, 1152, 160, 8]
    #biases = tf.get_variable(name = 'biases', 
    #                        shape = (1, self.caps_units, self.caps_dim)) # [1, 10, 16]
	
    inputs_tiled = tf.tile(inputs_tile, [1, 1, self.caps_units * self.caps_dim, 1], 
                            name = 'inputs_tiled') # [?, 1152, 160, 8]

    # inputs_tiled.shape = [?, 1152, 160, 8]
    # W.shape            = [1, 1152, 160, 8]

    Wu = tf.multiply(inputs_tiled, W, name = 'W_dot_u') # [?, 1152, 160, 8]
    #Wu = W * inputs_tiled # [?, 10, 1152, 16, 8]

    flatten_u_ji = tf.reduce_sum(Wu, axis = 3, keepdims = True, name = 'flatten_u_ji')# [?, 1152, 160, 1]

    u_ji = tf.reshape(flatten_u_ji, shape = [-1, self.input_caps_units, self.caps_units, self.caps_dim], 
                                    name = 'u_ji')			#[?, 1152, 10, 16]

    u_ji_stopped = tf.stop_gradient(u_ji, name = 'u_ji_stopped')        # [?, 1152, 10, 16]

    #u_ji = tf.reduce_sum(Wu, axis = 3, keepdims = False, name = 'u_ji') # [?, 1152, 160]
    
    #flatten_u_ji = tf.reduce_sum(Wu, axis = 4, keepdims = True, name = 'flatten_u_ji')# [?, 10, 1152, 16]

    #u_ji = tf.reshape(flatten_u_ji, shape = [-1, self.caps_units, self.caps_dim], 
    #                                name = 'u_ji')			#[?, 1152, 10, 16]

    #u_ji_stopped = tf.stop_gradient(u_ji, name = 'u_ji_stopped') #[?, 1152, 10, 16]

    v_j = []

    assert (self.batch_size % 5 == 0)
    for sample_k in range(self.batch_size):
      
      u_ji_k = u_ji[sample_k]                 # [1152, 10, 16]
      u_ji_stopped_k = u_ji_stopped[sample_k] # [1152, 10, 16]

      routing_logits = tf.zeros(shape = [self.input_caps_units, self.caps_units, 1], 
                                dtype = tf.float32, 
                                name  = 'rouring_logits') # [1152, 10, 1]
      
      for round_it in range(1, self.rounds + 1):
        with tf.variable_scope('round_' + str(round_it)):
          coupling_coeff = tf.nn.softmax(routing_logits, axis=1) # [1152, 10, 1]

          if round_it == self.rounds:
            # "For all but the first layer of capsules, the total input to a capsule s_j is a weighted sum over all
            #  'prediction vectors u_ji'"

            s_j = tf.multiply(coupling_coeff, u_ji_k) # [1152, 10, 1] . [1152, 10, 16] = [1152, 10, 16]
            s_j = tf.reduce_sum(s_j, axis = 0, keepdims = True)# + biases # [1, 10, 16]
            s_j = self.squash(s_j, axis = 2) # [1, 10, 16]

            v_j_k = tf.squeeze(s_j, axis = 0,name = 'v_j_k') # [10, 16]

            v_j.append(v_j_k)
            
          elif round_it < self.rounds:
            # coupling_coeff.shape = [1152, 10, 1]
            # u_ji_stopped_k.shape = [1152, 10, 16]
            s_j = tf.multiply(coupling_coeff, u_ji_stopped_k) # [1152, 10, 16]

            # "For all but the first layer of capsules, the total input to a capsule s_j is a weighted sum over all
            #  'prediction vectors u_ji'"

            s_j = tf.reduce_sum(s_j, axis = 0, keepdims = True)# + biases  # [1, 10, 16]
            v_j_k = self.squash(s_j) # [1, 10, 16]

            v_j_k_tiled = tf.tile(v_j_k, [self.input_caps_units, 1, 1]) # [1152, 10, 16]

            a_ij = tf.multiply(v_j_k_tiled, u_ji_stopped_k, name = 'a_ij') # [1152, 10, 16]

            agreement = tf.reduce_sum(a_ij, axis = 2, keepdims = True) # [1152, 10, 1]
            routing_logits += agreement
		
    return tf.stack(v_j) 