import tensorflow as tf
import sys
import os
import cifar10
import deep
import util as util
import numpy as np

FLAG_NEW_GRAPH = '-n'
FLAG_LOAD_GRAPH = '-l'

COMMAND_FLAG = None


# - Constants
# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256

img_size = 32   # CIFAR images are 32 by 32 pixels
# tuple with height and width used to reshape arrays
img_shape = (img_size, img_size, 3)

########################################################################
# Create the model
x = tf.placeholder(tf.float32, [None, 3072])

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 9])

# Build the graph for the deep net
y_conv, keep_prob = deep.deepnn(x)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# - Define Saver.
saver = tf.train.Saver()
# The saved files are often called checkpoints, they may be written at regular intervals during optimization.
save_dir = 'checkpoints/'
save_path = os.path.join(save_dir, 'best_validation')

# -- TensorFlow RUN
session = tf.Session()


########################################################################
def init(args):
    global COMMAND_FLAG

    COMMAND_FLAG = args[1]
    if COMMAND_FLAG == FLAG_NEW_GRAPH:
        # delete if save_dir existed and create a new one
        if tf.gfile.Exists(save_dir):
            tf.gfile.DeleteRecursively(save_dir)
        tf.gfile.MakeDirs(save_dir)
        # train and save graph
        train()
    elif COMMAND_FLAG == FLAG_LOAD_GRAPH:
        load_graph()

    session.close()


########################################################################
def train():
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    cucumber = cifar10.load_data()

    session.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = cucumber.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(session=session, feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(session=session, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(session=session, feed_dict={
        x: cucumber.test.images, y_: cucumber.test.labels, keep_prob: 1.0}))

    saver.save(sess=session, save_path=save_path)


########################################################################
def predict_cls(images, labels, cls_true):
    # Predicted Class
    y_pred = tf.nn.softmax(y_conv)
    # The class-number is the index of the largest element.
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    num_images = len(images)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0
    while i < num_images:
        j = min(i + batch_size, num_images)
        feed_dict = {x: images[i:j, :],
                     y_: labels[i:j, :],
                     keep_prob: 1.0}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    correct = (cls_true == cls_pred)
    return correct, cls_pred


def cls_accuracy(correct):
    correct_sum = correct.sum()
    acc = float(correct_sum) / len(correct)
    return acc, correct_sum


########################################################################
def load_graph():
    saver.restore(sess=session, save_path=save_path)
    print('weights loaded!')
    cucumber = cifar10.load_data()
    print('test accuracy %g' % accuracy.eval(session=session, feed_dict={
        x: cucumber.test.images, y_: cucumber.test.labels, keep_prob: 1.0}))

    # correct, cls_pred = predict_cls(images=cucumber.test.images,
    #                                 labels=cucumber.test.labels,
    #                                 cls_true=cucumber.test.cls)
    # acc, num_correct = cls_accuracy(correct)
    # num_images = len(correct)
    # msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    # print(msg.format(acc, num_correct, num_images))
    # util.plot_example_errors(cls_pred=cls_pred, correct=correct, data=cucumber, img_shape=img_shape)


########################################################################
def main(args):
    init(args)


########################################################################
# ENTRY POINT
if __name__ == '__main__':
    tf.app.run()
    main(sys.argv)