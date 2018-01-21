'''
Created on Dec 4, 2016

@author: Reihan
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition


def PCA(data, component):
    print("\n-----------------------------------------")
    print("implementing PCA....")
    pca = decomposition.PCA(component)
    pca.fit(data)
    trainIn = pca.transform(data)
    print("Done with PCA....\n")

    return trainIn