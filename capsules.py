import tensorflow as tf

class CapsNet:

	def __init__(	self, mode = 'conv'):
		self.mode = mode

	def layer(	self,
				inputs = None, 
				filters = None,
				kernel_size = 9,
				strides 	= (1, 1),
				padding 	= 'valid',
				activation 	= None,
				use_bias 	= True,
				trainable 	= True,
				caps_units 	= None, 
				caps_dim 	= None,
				rounds 		= 3,
				name        = None):

		if self.mode == 'conv2d':
			output = tf.layers.conv2d(	inputs 		= inputs,
										filters 	= filters,
										kernel_size = kernel_size,
										strides		= strides,
										padding 	= padding,
										activation	= activation,
										use_bias	= use_bias,
										trainable 	= trainable,
										name 		= name
										)

		elif self.mode == 'primary':
			conv  =  tf.layers.conv2d(	inputs 		= inputs,
										filters 	= caps_units * caps_dim,
										kernel_size = kernel_size,
										strides		= strides,
										padding 	= padding,
										activation	= activation,
										use_bias	= use_bias,
										trainable 	= trainable,
										name 		= name)

			grid_dim  = conv.shape[-2].value
			assert (grid_dim == conv.shape[-3].value)

			s 		  = tf.reshape(conv, [-1, grid_dim * grid_dim * caps_units, caps_dim])
			output    = self.squash(s, name = 'PrimaryCapsule_output')

		elif self.mode == 'digit':
			# inputs.shape = [?, 1152, 8]
			with tf.name_scope(name):
				inputs_expanded = tf.expand_dims(inputs, -1, name = 'inputs_expanded') # [?, 1152, 8, 1]
				inputs_tile     = tf.expand_dims(inputs_expanded,  2, name = 'inputs_tile') # [?, 1152, 1, 8, 1]

				self.caps_units = caps_units
				self.caps_dim   = caps_dim
				self.rounds		= rounds

				with tf.variable_scope('routing'):
					batch_size 		= tf.shape(inputs)[0]
					
					# "Initial logits b_ij are the log prior probabilities that capsule i shoould be coupled to capsule j"
					# b_ij => [#Capsules_i, #Capsules_j] :[1152x10]
					
					routing_logits 	= tf.zeros(shape = [batch_size, inputs.shape[1], caps_units, 1, 1], 
												dtype = tf.float32, 
												name  = 'rouring_logits') # [?, 1152, 10, 1, 1]
					#routing_logits 	= tf.zeros(shape = [batch_size, caps_units, 1, 1], dtype = tf.float32, 
					#																			   	    name  = 'rouring_logits')
					output = self.routing(inputs_tile, routing_logits, inputs.shape.as_list())

		return output		

	def squash(self, s, axis = -1, epsilon = 1e-9, name = 'squash'):
		with tf.name_scope(name):
			s_squared_norm  = tf.reduce_sum(tf.square(s), axis = axis, keepdims = True)
			squash_factor   = s_squared_norm / (1.0 + s_squared_norm)
			squash_unit_vec = s / tf.sqrt(s_squared_norm + epsilon)
			return  squash_factor * squash_unit_vec

	def routing(self, inputs_tile, routing_logits, inputs_shape):

		W_init  		= tf.random_normal( 
							shape  = (1, inputs_shape[1], self.caps_units * self.caps_dim, inputs_shape[-1], 1), #[1, 1152, 160, 8, 1]
							stddev = 0.01, 
							dtype  = tf.float32,
							name   = 'W_init')
		
		W 				= tf.Variable(W_init, name = 'W') #[1, 1152, 160, 8, 1]
		#print("W: ", W.shape)
		biases 			= tf.get_variable(name = 'biases', shape = (1, 1, self.caps_units, self.caps_dim, 1)) # [?, 1, 10, 16, 1]
	
		inputs_tiled    = tf.tile(inputs_tile, [1, 1, self.caps_units * self.caps_dim, 1, 1], 
									name = 'inputs_tiled') # [?, 1152, 160, 8, 1]

		Wu				 	= W * inputs_tiled # [?, 1152, 160, 8, 1]
		flatten_u_ji 		= tf.reduce_sum(Wu, axis = 3, keepdims = True, name = 'flatten_u_ji')# [?, 1152, 160, 1, 1]
		
		#print("flatten_u_ji : ", flatten_u_ji.shape)

		u_ji 	     	= tf.reshape(flatten_u_ji, shape = [-1, inputs_shape[1], self.caps_units, self.caps_dim, 1], 
												name = 'u_ji')			#[?, 1152, 10, 16, 1]

		u_ji_stopped	= tf.stop_gradient(u_ji, name = 'u_ji_stopped') #[?, 1152, 10, 16, 1]

		#print('u_ji: ', u_ji.shape)
		#print('u_ji_stopped: ', u_ji_stopped.shape)
		for round_it in range(1, self.rounds + 1):
			with tf.variable_scope('round_' + str(round_it)):
				coupling_coeff = tf.nn.softmax(routing_logits) # [?,1152,10,1,1]

				if round_it == self.rounds:
					# "For all but the first layer of capsules, the total input to a capsule s_j is a weighted sum over all
					#  'prediction vectors u_ji'"

					s_j = tf.multiply(coupling_coeff, u_ji) # [?,1152,10,1,1] . [?, 1152, 10, 16, 1] = [?, 1152,10,16,1]
					#print("s_ij 1: ", s_j.shape)
					s_j = tf.reduce_sum(s_j, axis = 1, keepdims = True) + biases # [?, 1,10,16,1]
					#print("s_ij 2: ", s_j.shape)
					v_j = self.squash(s_j)
					
					#print(v_j.shape, ' -> v_j shape last, round : ', round_it) # [?, 1, 10, 16, 1]
				elif round_it < self.rounds:
					s_j = tf.multiply(routing_logits, u_ji_stopped)
					#print("s_ij 1: ", s_j.shape) # [?, 1152,10,16,1]

					# "For all but the first layer of capsules, the total input to a capsule s_j is a weighted sum over all
					#  'prediction vectors u_ji'"

					s_j = tf.reduce_sum(s_j, axis = 1, keepdims = True) + biases  # [?,1,10,16,1]
					#print("s_ij 2: ", s_j.shape)
					v_j = self.squash(s_j) # [?,1,10,16,1]

					#print(v_j.shape, ' => v_j shape, round : ', round_it) 
					
					v_j_tiled = tf.tile(v_j, [1, u_ji_stopped.shape[1].value, 1, 1, 1]) # [?,1152,10,16,1]
					agreement = tf.reduce_sum(u_ji_stopped * v_j, axis = 3, keepdims = True) # [?,1152,10,1,1]
					#print(v_j_tiled.shape, ' => v_j_tiled')
					#print(agreement.shape, ' => agreement')
					
					routing_logits += agreement
		return v_j 