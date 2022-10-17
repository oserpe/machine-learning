import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from enum import Enum
from sklearn.model_selection import train_test_split

class ImageClasses(Enum):
    GRASS = 0
    SKY = 1
    COW = 2


def get_data_from_image(image_path, image_class):
    X = []
    y = []
    training_img = Image.open(image_path)

    for pixel in training_img.getdata():
        X.append(pixel)
        y.append(image_class.value)

    return X, y


if __name__ == "__main__":
    random_state = 1
    X = []
    y = []
    X, y += get_data_from_image(
        "ej2/data/partial_grass.jpg", ImageClasses.GRASS)
    X, y += get_data_from_image(
        "ej2/data/partial_sky.jpg", ImageClasses.SKY)
    X, y += get_data_from_image(
        "ej2/data/partial_cow.jpg", ImageClasses.COW)

    # Shuffle the data
    np.random.seed(random_state)
    np.random.shuffle(X)

    # Separate data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
