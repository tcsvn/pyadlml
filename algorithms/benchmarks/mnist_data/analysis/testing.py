from config import settings

import numpy as np
from ipywidgets import FloatProgress
from IPython.display import display
import pickle
import os.path

def evaluate_classification_accuracy(test_digits, predicted_labels):

    n_correct = 0
    n_false = 0

    n_correct_labels = [0,0,0,0,0,0,0,0,0,0]
    n_false_labels = [0,0,0,0,0,0,0,0,0,0]
    predictions = []
    for i in range(0, 10):
        predictions.append([0,0,0,0,0,0,0,0,0,0])


    i = 0
    for dig in test_digits:

        predictions[dig.label][predicted_labels[i]] += 1
        if dig.label == predicted_labels[i]:
            n_correct += 1
            n_correct_labels[dig.label] += 1
        else:
            n_false += 1
            n_false_labels[dig.label] += 1

        i += 1

    for i in range(0, 10):
        print("label " + str(i) + " : " + str(float(n_correct_labels[i]) / (float(n_correct_labels[i]) + float(n_false_labels[i]))), end = "")
        print("  ,  predictions : " + str(predictions[i]))

    return float(n_correct) / (float(n_correct) + float(n_false))
