import tensorflow as tf

class CapsLayer:
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
        # inputs_tile.shape = [?, 1152, 1, 8]
        # inputs_tiled.shape => [?, 1152, 160, 8]

        inputs_tile = tf.expand_dims(inputs,  2, name = 'inputs_tile')

        self.caps_units = caps_units
        self.caps_dim   = caps_dim
        self.rounds     = rounds

        with tf.variable_scope('routing'):
          self.batch_size = tf.shape(inputs)[0]

          # "Initial logits b_ij are the log prior probabilities that capsule i shoould be coupled to capsule j"
          # b_ij => [?, #Capsules_i, #Capsules_j] :[?, 1152, 10, 1]
          # routing_logits.shape = [?, 1152, 10, 1]

          routing_logits = tf.zeros(shape = [self.batch_size, self.input_caps_units, self.caps_units, 1],
                                    dtype = tf.float32,
                                    name  = 'rouring_logits')
          output = self.routing(inputs_tile, routing_logits)

    return output

  def squash(self, s, axis = -1, epsilon = 1e-7, name = 'squash'):
    with tf.name_scope(name):
      s_squared_norm  = tf.reduce_sum(tf.square(s), axis = axis, keepdims = True)
      squash_factor   = s_squared_norm / (1.0 + s_squared_norm)
      squash_unit_vec = s / tf.sqrt(s_squared_norm + epsilon)

      return  squash_factor * squash_unit_vec

  def routing(self, inputs_tile, routing_logits):
    # W.shape = [1, 1152, 160, 8]
    # inputs_tiled.shape = [?, 1152, 160, 8]

    W_init = tf.random_normal(
              shape  = (1, self.input_caps_units, self.caps_units*self.caps_dim, self.input_caps_dim), #[1, 1152, 160, 8]
              stddev = 0.01,
              dtype  = tf.float32,
              name   = 'W_init')

    W = tf.Variable(W_init, name = 'W')

    b_init = tf.zeros(shape = [1, self.input_caps_units, self.caps_units*self.caps_dim, 1],
                      dtype = tf.float32,
                      name  = 'b_init')

    biases = tf.Variable(b_init, name = 'bias')
    inputs_tiled = tf.tile(inputs_tile, [1, 1, self.caps_units*self.caps_dim, 1],
                            name = 'inputs_tiled')

    # W_dot_u.shape = [?, 1152, 160, 8]
    # flatten_u_ji.shape = [?, 1152, 160, 1]

    W_dot_u = tf.multiply(inputs_tiled, W, name = 'W_dot_u')
    flatten_u_ji = tf.reduce_sum(W_dot_u, axis = 3, keepdims = True, name = 'flatten_u_ji') + biases

    # u_ji.shape = [?, 1152, 10, 16]
    # u_ji_stopped.shape = [?, 1152, 10, 16]

    u_ji = tf.reshape(flatten_u_ji, shape = [-1, self.input_caps_units, self.caps_units, self.caps_dim],
                                    name = 'u_ji')
    u_ji_stopped = tf.stop_gradient(u_ji, name = 'u_ji_stopped')

    # round_it : [1, 2] if self.rounds = 3 (default)

    round_it = tf.constant(1)
    cond_rnd = lambda round_it, routing_logits, u_ji_stopped: tf.less(round_it, self.rounds)

    def rounting_loop(round_it, u_ji_stopped, routing_logits):
      with tf.variable_scope('rounting_loop'):
        # coupling_coeff.shape = [?, 1152, 10, 1]
        # u_ji_stopped.shape   = [?, 1152, 10, 16]

        coupling_coeff = tf.nn.softmax(routing_logits, axis=2, name='c_ij')

        # s_j_raw.shape = [?, 1152, 10, 16]
        # s_j.shape = [?, 1, 10, 16]
        # v_j.shape = [?, 1, 10, 16]

        s_j_raw = tf.multiply(coupling_coeff, u_ji_stopped, name = 's_j_raw')
        s_j = tf.reduce_sum(s_j_raw, axis = 1, keepdims = True, name = 's_j')
        v_j = self.squash(s_j, name='v_j')

        # u_ji_stopped.shape = [?, 1152, 10, 16]
        # v_j_tiled.shape = [1, 1152, 10, 16]
        # a_ij.shape =  [1, 1152, 10, 16]

        v_j_tiled = tf.tile(v_j, [1, self.input_caps_units, 1, 1], name = 'v_j_tiled')
        a_ij = tf.multiply(v_j_tiled, u_ji_stopped, name = 'a_ij')

        # agreement.shape = [?, 1152, 10, 1]
        # routing_logits.shape = [?, 1152, 10, 1]

        agreement = tf.reduce_sum(a_ij, axis = 3, keepdims = True)
        routing_logits += agreement
        round_it        = tf.add(round_it, 1)

      return round_it, u_ji_stopped, routing_logits



    *rest, routing_logits  = tf.while_loop(cond = cond_rnd,
                                           body = rounting_loop,
                                           loop_vars = [round_it,
                                                        u_ji_stopped,
                                                        routing_logits])
    # routing_logits.shape = [?, 1152, 10, 1]
    # coupling_coeff.shape = [?, 1152, 10, 1]
    # u_ji.shape = [?, 1152, 10, 16]

    # s_j_raw = multiply(coupling_coeff, u_ji)
    # s_j_raw.shape =[?, 1152, 10, 16]
    # s_j.shape = [?, 1, 10, 16]

    coupling_coeff = tf.nn.softmax(routing_logits, axis=2, name='c_ij')
    s_j_raw = tf.multiply(coupling_coeff, u_ji, name = 's_j_raw')
    s_j = tf.reduce_sum(s_j_raw, axis = 1, keepdims = True, name = 's_j')

    # v_j_raw.shape = [?, 1, 10, 16]
    # v_j.shape = [?, 10, 16]

    v_j_raw = self.squash(s_j, name='v_j_raw')
    v_j = tf.squeeze(v_j_raw, axis = 1, name='v_j')

    return v_j
