import tensorflow as tf
import numpy as np

from capsnet import CapsNet

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/')



def train(model):
	init = tf.global_variables_initializer()

	#saver = tf.train.Saver()

	n_epochs = 10
	batch_size = 50

	n_iter_train_per_epoch = mnist.train.num_examples // batch_size
	n_iter_valid_per_epoch = mnist.validation.num_examples // batch_size

	best_acc_val = 0.0

	print('\n\n Running CapsNet ...\n')
	with tf.Session() as sess:
		init.run()

		for epoch in range(n_epochs):
			loss_train_ep = []
			for it in range(1, n_iter_train_per_epoch + 1):
				X_batch, y_batch = mnist.train.next_batch(batch_size)

				_, loss_batch_train = sess.run(
								[model.train_op, model.batch_loss],
								feed_dict = {	model.X: X_batch.reshape([-1, 28, 28, 1]),
												model.y: y_batch,
												model.reconstruction: True})

				print("\rIter: {}/{} [{:.1f}%] loss : {:.5f}".format(
					it, n_iter_train_per_epoch, 100.0 * it / n_iter_train_per_epoch, loss_batch_train), end="")

				loss_train_ep.append(loss_batch_train)

			loss_train = np.mean(loss_train_ep)

			loss_val_ep = []
			acc_val_ep  = []

			for it in range(1, n_iter_valid_per_epoch + 1):
				X_batch, y_batch = mnist.validation.next_batch(batch_size)
				loss_batch_val, acc_batch_val = sess.run(
								[model.batch_loss, model.accuracy],
								feed_dict = {	model.X: X_batch.reshape([-1, 28, 28, 1]),
												model.y: y_batch})

				loss_val_ep.append(loss_batch_val)
				acc_val_ep.append(acc_batch_val)

				print("\rValidation ({:.1f}%)".format(100.0 * it / n_iter_valid_per_epoch), end=" "*15)

			loss_val = np.mean(loss_val_ep)
			acc_val  = np.mean(acc_val_ep)

			print("\repoch: {} loss_train: {:.4f}, loss_val: {:.4f}, valid_accuracy: {:.3f}% {}".format(
						epoch + 1, loss_train, loss_val, acc_val * 100.0, "(improved)" if acc_val > best_acc_val else ""))

			if acc_val > best_acc_val:
				best_acc_val = acc_val

if __name__ == '__main__':

	model = CapsNet()

	train(model)
