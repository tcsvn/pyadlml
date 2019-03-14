from config import settings

import matplotlib.pyplot as plt
import numpy as np
import itertools
from ipywidgets import FloatProgress
from IPython.display import display


def plot_digit(digit, display_progress = False):

    fig=plt.figure()
    ax=fig.add_subplot(111)

    f = FloatProgress(min=0, max=100)
    if display_progress:
        display(f)

    n_points = 0
    for curve in digit.curves:
        n_points += len(curve)

    i = 0
    for curve in digit.curves:
        x_points = []
        y_points = []
        for point in curve:
            x_points.append(point[0])
            y_points.append(point[1])
            f.value = 100.0*(float(i) / float(n_points))
            i += 1

        plt.plot(x_points, y_points, linewidth = 2.0)
    f.close()

    plt.axis([settings.IMAGE_PLOT_X_MIN, settings.IMAGE_PLOT_X_MAX, settings.IMAGE_PLOT_Y_MIN, settings.IMAGE_PLOT_Y_MAX])
    plt.show()


def plot_digits_heatmap(digits, display_progress = False):

    f = FloatProgress(min=0, max=100)
    if display_progress:
        display(f)

    plt.clf();
    _, axarr = plt.subplots(2, 5);

    for i in range(0, 2):
        for j in range(0, 5):

            n = 5*i + j

            x_points = []
            y_points = []
            for digit in digits:
                if digit.label == n:
                    for curve in digit.curves:
                        for point in curve:
                            x_points.append(point[0])
                            y_points.append(point[1])

            heatmap, xedges, yedges = np.histogram2d(x_points, y_points, bins=50);

            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]];

            axarr[i, j].imshow(np.rot90(heatmap), extent=extent);
            axarr[i, j].axis([settings.IMAGE_PLOT_X_MIN, settings.IMAGE_PLOT_X_MAX, settings.IMAGE_PLOT_Y_MIN, settings.IMAGE_PLOT_Y_MAX]);
            f.value += 10

    f.close()
    plt.show();



def plot_digit_observations(digit, centroids, n_observation_classes, display_progress = False):

    pen_down_label = n_observation_classes - settings.PEN_DOWN_LABEL_DELTA
    pen_up_label = n_observation_classes - settings.PEN_UP_LABEL_DELTA
    stop_label = n_observation_classes - settings.STOP_LABEL_DELTA

    fig=plt.figure()
    ax=fig.add_subplot(111)

    f = FloatProgress(min=0, max=100)
    if display_progress:
        display(f)

    curves = []
    current_curve = []
    for observation in digit.observations:
        if observation < pen_down_label:
            point = centroids[observation]
            current_curve.append(point)
        elif observation == pen_up_label:
            if len(current_curve) > 0:
                curves.append(current_curve)
            current_curve = []

    n_points = 0
    for curve in curves:
        n_points += len(curve)

    i = 0
    for curve in curves:
        x_points = []
        y_points = []
        for point in curve:
            x_points.append(point[0])
            y_points.append(point[1])
            f.value = 100.0*(float(i) / float(n_points))
            i += 1

        plt.plot(x_points, y_points, linewidth = 2.0)
    f.close()

    plt.axis([settings.IMAGE_PLOT_X_MIN, settings.IMAGE_PLOT_X_MAX, settings.IMAGE_PLOT_Y_MIN, settings.IMAGE_PLOT_Y_MAX])
    plt.show()


def plot_digit_samples(samples, display_progress = False):

    f = FloatProgress(min=0, max=100)
    if display_progress:
        display(f)

    plt.clf();
    _, axarr = plt.subplots(2, 5);

    for i in range(0, 2):
        for j in range(0, 5):

            n = 5*i + j
            n -= 1
            if n < 0:
                n = 9


            x_points = []
            y_points = []

            for curve in samples[n][0].curves:
                for point in curve:
                    x_points.append(point[0])
                    y_points.append(point[1])

            axarr[i, j].plot(x_points, y_points, linewidth = 2.0)
            #axarr[i, j].axis([settings.IMAGE_PLOT_X_MIN, settings.IMAGE_PLOT_X_MAX, settings.IMAGE_PLOT_Y_MIN, settings.IMAGE_PLOT_Y_MAX]);
            f.value += 10

    f.close()
    plt.show();


def plot_digit_samples_heatmap(samples, display_progress = False):

    f = FloatProgress(min=0, max=100)
    if display_progress:
        display(f)

    plt.clf();
    _, axarr = plt.subplots(2, 5);

    for i in range(0, 2):
        for j in range(0, 5):

            n = 5*i + j
            n -= 1
            if n < 0:
                n = 9

            x_points = []
            y_points = []
            for dig in samples[n]:
                for curve in dig.curves:
                    for point in curve:
                        x_points.append(point[0])
                        y_points.append(point[1])

            heatmap, xedges, yedges = np.histogram2d(x_points, y_points, bins=30);

            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]];

            axarr[i, j].imshow(np.rot90(heatmap), extent=extent);
            axarr[i, j].axis([settings.IMAGE_PLOT_X_MIN, settings.IMAGE_PLOT_X_MAX, settings.IMAGE_PLOT_Y_MIN, settings.IMAGE_PLOT_Y_MAX]);

            f.value += 10

    f.close()
    plt.show();
