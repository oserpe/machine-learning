import numpy as np
from matplotlib import pyplot as plt, test
from PIL import Image
from enum import Enum
from sklearn import svm, model_selection, utils


class ImageClasses(Enum):
    GRASS = 0
    SKY = 1
    COW = 2


def get_data_from_images(images_data):
    X = []
    y = []
    for image_path, image_class in images_data:
        training_img = Image.open(image_path)

        for pixel in training_img.getdata():
            X.append(pixel)
            y.append(image_class.value)

    return X, y


def test_smv(X_train, X_test, y_train, y_test, kernel, C=1, gamma='scale'):
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
    clf.fit(X_train, y_train)
    print(f"SVM {kernel} FITTED")

    return clf.score(X_test, y_test)


def test_different_smv_kernels(X_train, X_test, y_train, y_test):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    
    scores = []
    for kernel in kernels:
        print(f"Testing {kernel} kernel")
        scores.append(test_smv(X_train, X_test, y_train, y_test, kernel))
        print(f"Score for kernel {kernel}: {scores[-1]}")
    
    return scores


def test_different_smv_C(X_train, X_test, y_train, y_test):
    C_values = [0.1, 1, 10, 100, 1000]
    
    scores = []
    for C in C_values:
        print(f"Testing C={C}")
        scores.append(test_smv(X_train, X_test, y_train, y_test, 'rbf', C=C))
        print(f"Score for C={C}: {scores[-1]}")
    
    return scores


if __name__ == "__main__":
    random_state = 1
    X = []
    y = []

    images_directory = "machine-learning/ej2/data/"
    images_data = [
        (images_directory + "partial_grass.jpg", ImageClasses.GRASS),
        (images_directory + "partial_sky.jpg", ImageClasses.SKY),
        (images_directory + "partial_cow.jpg", ImageClasses.COW)]

    X, y = get_data_from_images(images_data)

    # Shuffle the data
    X, y = utils.shuffle(X, y, random_state=random_state)

    # Separate data into training and test data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=random_state)
    print("READY TO FIT")

    # Test different kernels
    # test_different_smv_kernels(X_train, X_test, y_train, y_test)

    # Test different C values
    test_different_smv_C(X_train, X_test, y_train, y_test)

