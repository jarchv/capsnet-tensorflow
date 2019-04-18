import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from capsnet import CapsNet

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/')
batch_size = 64


def train(model, restore = False):
	init = tf.global_variables_initializer()
	n_epochs = 50
	

	n_iter_train_per_epoch = mnist.train.num_examples // batch_size
	n_iter_valid_per_epoch = mnist.validation.num_examples // batch_size

	best_loss_val = np.infty

	checkpoint_file = './capsnet'

	saver = tf.train.Saver()
	with tf.Session() as sess:
		writer = tf.summary.FileWriter("output", sess.graph)

		if restore and tf.train.checkpoint_exists(checkpoint_file):
			saver.restore(sess, checkpoint_file)
		else:
			init.run()

		print('\n\nRunning CapsNet ...\n')
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

				print("\rValidation ({:.1f}%)".format(100.0 * it / n_iter_valid_per_epoch), end=" "*30)

			loss_val = np.mean(loss_val_ep)
			acc_val  = np.mean(acc_val_ep)

			print("\repoch: {} loss_train: {:.5f}, loss_val: {:.5f}, valid_accuracy: {:.4f}% {}".format(
						epoch + 1, loss_train, loss_val, acc_val * 100.0, "(improved)" if loss_val < best_loss_val else ""))

			if loss_val < best_loss_val:
				save_file = saver.save(sess, checkpoint_file)
				best_loss_val = loss_val

		writer.close()

def test(model):
	checkpoint_file = './capsnet'
	saver = tf.train.Saver()
	n_iter_test_per_epoch = mnist.test.num_examples // batch_size

	loss_test_ep = []
	acc_test_ep  = []

	with tf.Session() as sess:
		saver.restore(sess, checkpoint_file)

		print('\n\nTest\n')
		for it in range(1, n_iter_test_per_epoch + 1):
			X_batch, y_batch = mnist.test.next_batch(batch_size)
			loss_batch_test, acc_batch_test = sess.run(
								[model.batch_loss, model.accuracy],
								feed_dict = {	model.X: X_batch.reshape([-1, 28, 28, 1]),
												model.y: y_batch})

			loss_test_ep.append(loss_batch_test)
			acc_test_ep.append(acc_batch_test)
			print("\rTesting .. ({:.1f}%)".format(100.0 * it / n_iter_test_per_epoch), end=" "*30)	

		loss_test = np.mean(loss_test_ep)
		acc_test  = np.mean(acc_test_ep)

		print("\r(Testing) accuracy: {:.2f}%, loss: {:.4f}".format(acc_test*100.0, loss_test))

def reconstruction(model, num_samples):
	checkpoint_file = './capsnet'
	saver = tf.train.Saver()

	samples_imgs = mnist.test.images[:num_samples].reshape([-1, 28, 28, 1])

	with tf.Session() as sess:
		saver.restore(sess, checkpoint_file)

		decoder_output, y_pred_value = sess.run(
			[model.decoder_output, model.y_pred],
			feed_dict = {	model.X: samples_imgs,
							model.y: np.array([], dtype = np.int64)})


	samples_imgs = samples_imgs.reshape([-1, 28, 28])
	reconstructions_imgs = decoder_output.reshape([-1, 28, 28])	

	plt.figure(figsize = (num_samples * 2, 4))

	for img_idx in range(num_samples):
		plt.subplot(2, num_samples, img_idx + 1)
		plt.imshow(samples_imgs[img_idx], cmap='gray')
		plt.title("Input: " + str(mnist.test.labels[img_idx]))
		plt.axis("off")

	#plt.show()
	for img_idx in range(num_samples):
		plt.subplot(2, num_samples, num_samples + img_idx + 1)
		plt.imshow(reconstructions_imgs[img_idx], cmap='gray')
		plt.title("Output: " + str(y_pred_value[img_idx]))
		plt.axis("off")

	plt.show()
if __name__ == '__main__':

	model = CapsNet()
	#train(model, True)
	test(model)
	#reconstruction(model, 5)
