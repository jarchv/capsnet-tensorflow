import tensorflow as tf
import numpy as np

from capsules import CapsNet as caps

class CapsNet:
  def __init__(self, 
              mode = 'train',
              classes = 10, 
              m_plus  = 0.9,
              m_minus = 0.1,
              lambda_ = 0.5,   
              alpha   = 0.0005,
              rounds  = 3) :  				

    self.X = tf.placeholder(shape = [None, 28, 28, 1], 
                            dtype = tf.float32, 
                            name  = 'X')
		
    self.classes = classes
    self.m_plus  = m_plus
    self.m_minus = m_minus
    self.lambda_ = lambda_
    self.alpha   = alpha
    self.rounds  = rounds
    self.build_model()

  def build_model(self):
    self.X_pad     = tf.image.resize_image_with_crop_or_pad(self.X, 32, 32)
    self.X_cropped = tf.random_crop(self.X_pad, [tf.shape(self.X)[0], 28, 28, 1])
		
    with tf.name_scope('Conv1'):
      self.conv1_output   = caps(mode = 'conv2d').layer(inputs      = self.X_cropped,
                                                        filters     = 256, 
                                                        kernel_size = 9, 
                                                        activation  = tf.nn.relu,
                                                        name        = 'Conv1'
                                                        )

    with tf.name_scope('PrimaryCapsule'):
      self.prycaps_output = caps(mode = 'primary').layer(inputs     = self.conv1_output,
                                                        kernel_size = 9,
                                                        strides     = 2,
                                                        activation  = tf.nn.relu,
                                                        caps_units  = 32,
                                                        caps_dim    = 8,
                                                        name        = 'PrimaryCaps'
                                                        )
    
    # digcaps_output : [?, 10, 16]
    with tf.name_scope('DigitCapsule'):
      self.digcaps_output = caps(mode = 'digit').layer(inputs     = self.prycaps_output,
                                                      caps_units  = self.classes,
                                                      caps_dim    = 16,
                                                      rounds      = self.rounds,
                                                      name        = 'DigitCaps'
                                                      )

    with tf.name_scope('Masking'):
      self.v_j_length = self.get_length(self.digcaps_output, axis = -1) # [?, 10]
      self.y = tf.placeholder_with_default(np.array([-1], dtype=np.int64), shape = [None], name = 'y')
      
      with tf.variable_scope('Masking'):
        self.batch_sum_loss_rec = self.reconstruction_loss()

    with tf.name_scope('Total_Loss'):
      self.batch_loss = self.margin_loss() + self.batch_sum_loss_rec * self.alpha	

    with tf.variable_scope('Accuracy'):
      self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_pred), tf.float32), 
                                    name = 'accuracy_mean')

    with tf.name_scope('Training'):
      with tf.variable_scope('Train'):
        starter_learning_rate = 0.001
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1.0, 0.9, staircase=True)

        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        # decayed_learning_rate = 0.001 * 0.9 ^ (global_step / 1.0)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op  = self.optimizer.minimize(self.batch_loss, global_step=global_step, name = 'train_op')

  def get_length(self, v_j, axis = -1, keepdims = False, name = None):
    with tf.variable_scope('lenght'):
      v_j_squared = tf.reduce_sum(tf.square(v_j) + 1e-7, 
                      axis      = axis,
                      keepdims  = keepdims)
      return tf.sqrt(v_j_squared, name = 'v_j_length')

  def margin_loss(self):
    self.present_e  = tf.square(tf.maximum(0.0,  self.m_plus  - tf.reshape(self.v_j_length, [-1, self.classes])),  name = 'present_error')
    self.absent_e 	= tf.square(tf.maximum(0.0, -self.m_minus + tf.reshape(self.v_j_length, [-1, self.classes])), name = 'absent_error')

    T = tf.one_hot(self.y, depth = self.classes, name = 'T')
    L = tf.add(T * self.present_e, self.lambda_ * (1.0 - T) * self.absent_e)  # [?, 10]

    margin_loss = tf.reduce_sum(L, axis = 1, name = 'margin_loss') # [?,]
    batch_loss  = tf.reduce_mean(margin_loss, name = 'batch_loss', axis=0) # []

    return batch_loss

  def reconstruction_loss(self):
    self.y_pred = tf.argmax(self.v_j_length, axis = 1, name = 'y_pred') # [?,]

    self.reconstruction = tf.placeholder_with_default(False, shape = (), name = 'label_mask')		
    self.label_to_mask  = tf.cond(self.reconstruction, lambda: self.y, lambda: self.y_pred, name = 'label_to_mask')

    self.mask = tf.one_hot(self.label_to_mask, depth = self.classes, name = 'mask')
    self.mask_reshaped = tf.reshape(self.mask, shape = [-1, self.classes, 1])

    # digcaps_output.shape : [?, 10, 16]
    # mask_reshaped.shape  : [?, 10,  1]
    self.caps_output_masked = tf.multiply(self.digcaps_output, self.mask_reshaped, name = 'caps_output_masked') # [?, 10, 16]
    self.decoder_input = tf.reshape(self.caps_output_masked, shape = [-1, self.classes * self.digcaps_output.shape[-1].value]) # [?, 160]

    #print("self.digcaps_output : ", self.digcaps_output.shape)
    #print("self.mask_reshaped : ", self.mask_reshaped.shape)
    #print("decoder_input : ", self.decoder_input.shape)

    with tf.name_scope('Decoder'):
      fc1 = tf.layers.dense(inputs     = self.decoder_input, 
                            units      = 512,
                            activation = tf.nn.relu,
                            name       = 'fc1')

      fc2 = tf.layers.dense(inputs      = fc1,
                            units       = 1024,
                            activation  = tf.nn.relu,
                            name        = 'fc2')

      self.decoder_output = tf.layers.dense(inputs      = fc2,
                                            units       = 784,
                                            activation  = tf.nn.sigmoid,
                                            name        = 'decoder_output')

    self.flatten_X = tf.reshape(self.X_cropped, shape = [-1, 784], name = 'flatten_X')

    self.squared_diff = tf.square(self.flatten_X - self.decoder_output, name = 'squared_diff')
    self.sum_loss = tf.reduce_sum(self.squared_diff , name = 'reconstruction_loss', axis=-1)
    self.batch_sum_loss = tf.reduce_mean(self.sum_loss, name = 'reconstruction_batch_loss')

    return self.batch_sum_loss