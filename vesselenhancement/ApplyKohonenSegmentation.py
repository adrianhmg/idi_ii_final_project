
import numpy as np
from PIL import Image as IM
from algorithms.kohonen import kohonen_trainning


def replace_pixels_centroids(class_centroids, classes_for_data, n_data):
    for i in range(0, n_data):
        classes_for_data[i] = np.uint8(class_centroids[classes_for_data[i]])

    return classes_for_data


def applyKohonen(I,a,b):
    np.random.seed(50)
    I = np.array(I)
    I = IM.fromarray(np.uint8(I))

    #print(I.size[0], I.size[1], I.mode, I.format)

    M = np.asarray(I)

    x = M[:, :, 0]
    x = x.flatten()

    y = M[:, :, 1]
    y = y.flatten()

    z = M[:, :, 2]
    z = z.flatten()

    n_data = I.size[0] * I.size[1]

    lim_inf=1.5
    lim_max=b

    k=5
    #k=35
    #print(lim_max)
    centroids = np.random.randint(lim_inf,lim_max,(k,3))

    presc=35

    # class_centroids,classes_for_data = kk_means(x,y,z,centroids,k)
    class_centroids, classes_for_data = kohonen_trainning(x, y, z, centroids, k, presc, n_data)

    q = centroids
    s = []
    for i in range(0, len(q)):
        s.append(sum(q[i]))

    mean_s = a

    for i in range(0, len(q)):
        if s[i] < mean_s:
            q[i] = [0, 0, 0]
        else:
            q[i] = [255, 255, 255]


    new_replace_array = np.array(replace_pixels_centroids(q, classes_for_data, n_data))
    new_replace_array = np.reshape(new_replace_array, (I.size[1], I.size[0], 3))
    new_replace_array = np.array(new_replace_array)

    return new_replace_array

