import numpy as np
from matplotlib import pyplot as plt, test
from PIL import Image, ImageColor
from enum import Enum
from sklearn import svm, model_selection, utils
import seaborn as sns


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
    print(f"SVM {kernel} started training")
    clf.fit(X_train, y_train)
    print(f"SVM {kernel} trained")

    return model_selection.cross_val_score(clf, X_test, y_test, cv=5).mean()


def test_different_smv_kernels(X_train, X_test, y_train, y_test):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    
    scores = []
    for kernel in kernels:
        print(f"Testing {kernel} kernel")
        scores.append(test_smv(X_train, X_test, y_train, y_test, kernel))
        print("%0.6f accuracy with a standard deviation of %0.6f" % (scores[-1].mean(), scores[-1].std()))
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


def svm_image_classification(X_train, y_train, image_path):
    print("Creating svm classificator")
    clf = svm.SVC(kernel='rbf', C=10, gamma='scale')
    clf.fit(X_train, y_train)

    print("Starting image classification")
    img = Image.open(image_path)
    # Colors for each class: green, lightblue and brown
    colors = [(0, 255, 0), (0, 255, 255), (165, 42, 42)]

    # Classify pixels
    pixels_class = clf.predict(img.getdata())
    # Get color of each pixel by their predicted class
    pixels_new_color = [colors[pixel_class] for pixel_class in pixels_class]

    print("Creating new image")
    new_img = Image.new(img.mode, img.size)
    new_img.putdata(pixels_new_color)

    new_img.show()


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

    # Separate data into training and test data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=random_state)

    # --- EXERCISES ---

    # Test different kernels
    test_different_smv_kernels(X_train, X_test, y_train, y_test)

    # Test different C values
    # test_different_smv_C(X_train, X_test, y_train, y_test)

    # Predict image pixels classes
    # svm_image_classification(X_train, y_train, images_directory + "image3.jpg")
