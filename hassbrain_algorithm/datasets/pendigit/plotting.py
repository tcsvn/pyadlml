import matplotlib.pyplot as plt
import numpy as np

def plotUniPenData(points):
    xs = []
    ys = []
    ind = 0
    x = []
    y = []
    if isinstance(points, list):
        for point in points:
            if (point[0] == -1):
                xs.append(x)
                ys.append(y)
                x = []
                y = []
                ind += 1
                continue
            x.append(point[0])
            y.append(point[1])
        for i in range(ind):
            plt.plot(xs[i], ys[i])
        plt.show()
    else:
        for (index, point) in enumerate(points):
            if (point[0] == -1):
                xs.append(x)
                ys.append(y)
                x = []
                y = []
                ind += 1
                continue
            x.append(point[0])
            y.append(point[1])
        for i in range(ind):
            plt.plot(xs[i], ys[i])
        plt.show()

def plotVoronoid(data, estimator):
    x_min, x_max = np.min(data), np.max(data)
    y_min, y_max = np.min(data), np.max(data)

    h = (x_max - x_min) / 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    # Plot the centroids as a white X
    centroids = estimator.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('Clustering Bins for Digit data')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
