import numpy as np
import matplotlib.pyplot as plt


class DataLoader:
    def class_weights(COUNT_PNEUMONIA, COUNT_NORMAL, TRAIN_IMG_COUNT):
        initial_bias = np.log([COUNT_PNEUMONIA / COUNT_NORMAL])
        print(initial_bias)
        weight_for_0 = (1 / COUNT_NORMAL) * (TRAIN_IMG_COUNT) / 2.0
        weight_for_1 = (1 / COUNT_PNEUMONIA) * (TRAIN_IMG_COUNT) / 2.0

        class_weight = {0: weight_for_0, 1: weight_for_1}
        return class_weight

    def get_generator(datagen, directory, img_width, img_height, BATCH_SIZE):
        generator = datagen.flow_from_directory(
            directory,
            target_size=(img_width, img_height),
            batch_size=BATCH_SIZE,
            class_mode="binary",
        )
        return generator

    def show_batch(image_batch, label_batch, BATCH_SIZE):
        plt.figure(figsize=(10, 10))
        for n in range(BATCH_SIZE):
            ax = plt.subplot(5, 5, n + 1)
            plt.imshow(image_batch[n])
            if label_batch[n]:
                plt.title("PNEUMONIA")
            else:
                plt.title("NORMAL")
            plt.axis("off")
            plt.savefig("image_batch.png")
