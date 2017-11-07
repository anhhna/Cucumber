import matplotlib.pyplot as plt


# Func: plotting/showing images
def plot_images(images, cls_true, class_names,
                cls_pred=None, img_shape=None):
    assert len(images) == len(cls_true) == 9

    # create figure with 2x2 sub-plots
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape))

        # Name of the true class.
        cls_true_name = class_names[cls_true[i]]

        # true and predicted class
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]

            xlabel = "True: {0}, Pred: {1}".format(cls_true_name, cls_pred_name)

        ax.set_xlabel(xlabel)

        # remove ticks from plot
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# Helper-function to plot example errors
def plot_example_errors(cls_pred, correct, data, img_shape):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9], class_names=data.class_names,
                cls_pred=cls_pred[0:9], img_shape=img_shape)
    plt.show()