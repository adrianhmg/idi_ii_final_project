import numpy as np


def kohonen_trainning(x, y, z, centroids, k, presc,n_data):
    centroids_ant = np.zeros((k, 3))

    it = 0
    N = 2
    dist_cent = [100000]

    # while it<5:
    while max(dist_cent) > presc:
        classes_for_data = []
        centroids_ant = centroids.copy()
        it += 1
        # print("it",it,"\n centroids: \n",centroids)
        distClasses = []
        ## Iteramos desde cada punto del data, y luego sus centroids
        for i in range(0, n_data):
            dist = []
            for l in range(0, k):
                dist.append(
                    ((x[i] - centroids[l][0]) ** 2) + ((y[i] - centroids[l][1]) ** 2) + ((z[i] - centroids[l][2]) ** 2))
            dist = np.asarray(dist)
            min_ind = dist.argmin(axis=0)
            # Asignamos en los centroids el nuevo valor
            centroids[min_ind][0] = round((x[i] + (N - 1) * (centroids[min_ind][0])) / N, 2)
            centroids[min_ind][1] = round((y[i] + (N - 1) * (centroids[min_ind][1])) / N, 2)
            centroids[min_ind][2] = round((z[i] + (N - 1) * (centroids[min_ind][2])) / N, 2)
            # print(centroids)
            classes_for_data.append(min_ind)
        ## Calculamos precision
        dist_cent = []
        for n in range(0, k):
            dist_cent.append(
                ((centroids_ant[n][0] - centroids[n][0]) ** 2) + ((centroids_ant[n][1] - centroids[n][1]) ** 2) + (
                            (centroids_ant[n][2] - centroids[n][2]) ** 2))
        N += 1
        #print("precision", max(dist_cent))

    return centroids, classes_for_data