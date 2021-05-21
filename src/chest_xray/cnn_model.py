from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


class Model:
    def make_model(input_shape, output_shape):
        cnn = models.Sequential(
            [
                layers.Conv2D(
                    32,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="valid",
                    activation="relu",
                    input_shape=input_shape,
                ),
                layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
                layers.Conv2D(
                    64,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="valid",
                ),
                layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
                layers.Conv2D(
                    64,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="valid",
                ),
                layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(output_shape, activation="sigmoid"),
            ]
        )
        return cnn

    def make_graph(history, metrics):
        fig, ax = plt.subplots(1, 4, figsize=(20, 3))
        ax = ax.ravel()
        for i, met in enumerate(metrics):
            ax[i].plot(history.history[met])
            ax[i].plot(history.history["val_" + met])
            ax[i].set_title(f"Model {met}")
            ax[i].set_xlabel("epochs")
            ax[i].set_ylabel(met)
            ax[i].legend(["train", "val"])
        plt.savefig("training_history.png")
