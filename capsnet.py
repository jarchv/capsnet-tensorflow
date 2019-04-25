import tensorflow as tf

from capsules import CapsNet as caps

class CapsNet:
	def __init__(self, 	mode = 'train',
						classes = 10, 
						m_plus  = 0.9,
						m_minus = 0.1,
						lambda_ = 0.5,
						alpha   = 0.0005,
						rounds  = 3) :

		tf.reset_default_graph()

		self.X = tf.placeholder(shape = [None, 28, 28, 1], dtype = tf.float32, name = 'X')
		
		self.classes = classes
		self.m_plus  = m_plus
		self.m_minus = m_minus
		self.lambda_ = lambda_
		self.alpha   = alpha
		self.rounds  = rounds

		self.build_model()

	def build_model(self):

		self.X_pad      = tf.image.resize_image_with_crop_or_pad(self.X, 32, 32)
		self.X_cropped  = tf.random_crop(self.X_pad, [tf.shape(self.X)[0], 28, 28, 1])
		self.Conv1 		= caps('conv2d').layer(	 inputs = self.X_cropped,
												filters = 256, 
												kernel_size = 9, 
												activation = tf.nn.relu,
												name   = 'Conv1'
												)

		self.PrimaryCaps = caps('primary').layer(inputs 	= self.Conv1,
												kernel_size = 9,
												strides 	= 2,
												activation  = tf.nn.relu,
												caps_units  = 32,
												caps_dim    = 8,
												name 		= 'PrimaryCaps'
												)

		self.DigitCaps 	= caps('digit').layer(	inputs = self.PrimaryCaps,
												caps_units 	= self.classes,
												caps_dim  	= 16,
												rounds		= self.rounds,
												name		= 'DigitCaps')

		self.y = tf.placeholder(shape = [None], dtype = tf.int64, name = 'y')

		with tf.variable_scope('masking'):
			self.v_j_length = self.safe_length(self.DigitCaps, axis = -2)
			self.batch_loss = self.margin_loss() + self.reconstruction_loss() * self.alpha	

		with tf.variable_scope('Accuracy'):
			self.correct  = tf.equal(self.y, self.y_pred, name = 'correct')
			self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32), name = 'accuracy_mean')

		with tf.variable_scope('Train'):
			self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
			self.train_op  = self.optimizer.minimize(self.batch_loss, name = 'train_op')

	def safe_length(self, v_j, axis, epsilon = 1e-9, keepdims = False, name = None):
		with tf.variable_scope('safe_lenght'):
			v_j_squared =  tf.reduce_sum(	tf.square(v_j), 
											axis 	 = axis,
											keepdims = keepdims)
			return tf.sqrt(v_j_squared + epsilon, name = 'v_j_length')

	def margin_loss(self):

		self.present_e_raw  = tf.square(tf.maximum(0.0, self.m_plus - self.v_j_length),  name = 'present_error_raw')
		self.absent_e_raw 	= tf.square(tf.maximum(0.0, self.v_j_length - self.m_minus), name = 'absent_error_raw')

		self.present_e = tf.reshape(self.present_e_raw, shape = (-1, self.classes))
		self.absent_e  = tf.reshape(self.absent_e_raw , shape = (-1, self.classes))

		
		T = tf.one_hot(self.y, depth = self.classes, name = 'T')


		L = tf.add(T * self.present_e, self.lambda_ * (1.0 - T) * self.absent_e) 

		margin_loss = tf.reduce_sum(L, axis = 1, name = 'margin_loss')
		
		batch_loss  = tf.reduce_mean(margin_loss, name = 'batch_loss')

		return batch_loss

	def reconstruction_loss(self):
		#print(self.v_j_length.shape, " v_j_length")
		self.argmax_target = tf.argmax(self.v_j_length, axis = 1, name = 'argmax_target')
		#print(self.argmax_target.shape, " argmax")
		self.y_pred        = tf.squeeze(self.argmax_target, axis = 1, name = 'y_pred')

		self.reconstruction = tf.placeholder_with_default(False, shape = (), name = 'label_mask')		
		self.label_to_mask  = tf.cond(self.reconstruction, lambda: self.y, lambda: self.y_pred, name = 'label_to_mask')

		self.mask 				= tf.one_hot(self.label_to_mask, depth = self.classes, name = 'mask')
		self.mask_reshaped  	= tf.reshape(self.mask, shape = [-1, self.classes, 1, 1])
		#print(self.DigitCaps.shape, ' self.DigitCaps')
		self.caps_output_masked = tf.multiply(self.DigitCaps, self.mask_reshaped, name = 'caps_output_masked') # [?, 16, 10, 1]
		#print(self.caps_output_masked.shape, ' self.caps_output_masked')
		self.decoder_input = tf.reshape(self.caps_output_masked, shape = [-1, self.classes * self.caps_output_masked.shape[-2].value]) # [?, 160]


		with tf.name_scope('Decoder'):
			fc1 = tf.layers.dense(  inputs = self.decoder_input, 
									units  = 512,
									activation = tf.nn.relu,
									name 	   = 'fc1')

			fc2 = tf.layers.dense(	inputs = fc1,
									units  = 1024,
									activation  = tf.nn.relu,
									name 		= 'fc2')

			self.decoder_output = tf.layers.dense( 	inputs = fc2,
													units  = 784,
													activation  = tf.nn.sigmoid,
													name  		= 'decoder_output')

		self.flatten_X = tf.reshape(self.X_cropped, shape = [-1, 784], name = 'flatten_X')

		self.squared_diff = tf.square(self.flatten_X - self.decoder_output, name = 'squared_diff')
		self.reconstruction_batch_loss = tf.reduce_mean(self.squared_diff, name = 'reconstruction_loss')

		return self.reconstruction_batch_loss