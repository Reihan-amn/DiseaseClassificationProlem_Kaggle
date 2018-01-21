import matplotlib.pyplot as plt
import numpy as np
import DataIOFactory
import scipy as sp
import sklearn
from sklearn.neighbors import KNeighborsRegressor

from sklearn import decomposition
from sklearn.feature_selection import VarianceThreshold 
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colors
import six
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
import math
from operator import itemgetter
from sklearn.metrics import precision_recall_fscore_support
from matplotlib import colors
from PIL.ImageColor import getcolor
from DataIOFactory import balancingData

'''reading data from csv file and only getting first 100,000 instances for this experiment'''

data2DMat = DataIOFactory.getDataMatrixFromCSV("./DeathRecords.csv")
print('type: ', type(data2DMat),' shape: ',data2DMat.shape)




''' extracting feature labels and class label'''

columns_name = np.array(data2DMat[0])#first rows has the column names
features_label =  np.delete(columns_name, [0,24], axis = None) #removing the id and class column
data2DMat = np.delete(data2DMat, (0), axis=0) 
print('whole data shape  ', data2DMat.shape)




'''shuffling the data'''
shuffeledMat = DataIOFactory.matrixShuffling(data2DMat)



'''
this dataset is about mortality and death could have different causes.
in this part, we are only taking the data instances that are dealing with a type of disease based on the column: Icd10
imbalanced data
'''
diseaseData = DataIOFactory.pickDiseasetargets(shuffeledMat, columns_name)
print("lenght of disease data" ,len(diseaseData))

'''extracting a balanced number of classes form data
balanced data
'''
balancedData =  DataIOFactory.balancingData(diseaseData,12, columns_name)
print('balancedData: ' , balancedData.shape)


'''Column 24, icd10 is our class and the rest are features '''
clss = diseaseData[:, 24]
print("class lenght:" ,len(clss))
clssa = DataIOFactory.classDiscreteValueConverToDecimal(clss)
classes = clssa[:,None] #clss is just 1D array, we have to convert it to 2D array




'''select the columns as input features - all columns but 0 and 24'''
features =  np.delete(diseaseData, [0,24], axis = 1)




'''converting descrete featuers to digit'''
featureMatrix = DataIOFactory.featureDiscreteToZeroOne(features, 5, 14, 17, 19, 20)  


'''converting all features to float64'''
floatFeaturesMatrix = DataIOFactory.matrixConvertToFloat(featureMatrix)



'''normalizing the data'''
# inputNormalizedMatrix = DataIOFactory.RowToColumnTranspositionMatrix(DataIOFactory.normalizingFeatures(floatFeaturesMatrix))


    
'''spliting the data into test and train and validation set based on different portions'''
trainIn, trainOut, validationIn, validationOut, testIn, testOut = DataIOFactory.dataSplitFactory(floatFeaturesMatrix, classes, 0.8, 0.1, 0.1)

print('trainIn' ,len(trainIn))
print('trainout', len(trainOut))
print(trainIn)
# print(validationOut)
      
# pca = decomposition.PCA(n_components=3)
# pca.fit(trainIn)
# print("before PCA")
# print(trainIn)
# trainIn = pca.transform(trainIn)
# print("after PCA")
# print(trainIn)
# --------------low variance----
# print(trainIn)
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# print(sel.fit_transform(trainIn))
# #------------------------ 

'''chi2 feature selection'''
X_new = SelectKBest(chi2, k = 'all').fit(trainIn, trainOut) 
print('scores')
# print(type(X_new.scores_))


print('feature size' , len(features_label))
print('score size' , len(X_new.scores_))
dtype = [('label', 'S10'), ('score', float)]
feat_score = []
for i in range(0, len(features_label)):
    temp = []
#     print('features' , features_label[i], '        score: '  ,X_new.scores_[i] )
    temp.append(features_label[i])
    temp.append(X_new.scores_[i])
#     print(temp)
    feat_score.append(temp)
    
# print(feat_score)
sortedList = sorted(feat_score, key=(lambda x: x[1]), reverse = True)
# print(sortedList)

'''classification Knearest based on all features'''
neigh = KNeighborsRegressor(n_neighbors=5)
neigh.fit(trainIn, trainOut)
# print("for this input:")
# print()
print("predicted output for validation is")
print()
# print(neigh.predict(validationIn))
valid_pred =  neigh.predict(validationIn)
'''classification knearest based on top features in chi2'''
# print('valid_out ' , validationOut.shape , '  valid_pred  ' , valid_pred.shape , ' class-label  ' ,  columns_name.shape )

# print(floatFeaturesMatrix)


valid_pred = DataIOFactory.roundingNumbers(valid_pred)

# for i in range(0, len(valid_pred)):
#     print('predict: ' , valid_pred[i], '  real: ' , validationOut[i] )

n_real = validationOut.flatten()
n_predict = valid_pred.flatten()

'''confusion matrix'''
nf_mat = confusion_matrix(n_real, n_predict)
x = np.array(nf_mat)
# print(x)

print(precision_recall_fscore_support(n_real, n_predict, average='macro'))


# precision_recall_fscore_support(validationOut, valid_pred, average='macro')
# plot_confusion_matrix(validationOut, valid_pred,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm)

# nwLis = feat_score.sort(key=itemgetter(1))   
# print('all here: ' , feat_score.sort(key=itemgetter(1)) )
# sortedScore = np.sort(score)
# print('sorted   : ' , sortedScore)



'''plotting the chart of scoring the features'''   
# def getColorNames():
#     
#     colorNames = []
#     for color in colors.cnames:    
#         colorNames.append(color)
#     
#     return colorNames
# 
# colors = getColorNames()
# counter  = 0
# for score in X_new.scores_:
#     if(score > 10):
#         print('{:>34}'.format( features_label[counter]))
#         print('{:>34}  '.format( score))
#     '''Plot a graph'''    
#     plt.bar(counter, score, color = colors[counter])
#     counter +=1
# 
#  
# plt.ylabel('Scores')
# plt.title('Scores calculated by Chi-Test')
# plt.legend(features_label, bbox_to_anchor=(0., 0.8, 1., .102), loc=3,ncol=5, mode="expand", borderaxespad=0.)
# plt.show()
#     
#     

# adding column to a matrix
# features_matrix = np.hstack((featuresId, inputNormalizedMatrix))


# selecting specific column
# print('feature size' , features.shape)
# idx_IN_columns = [1, 2, 3, 4, 5, 6, 7,  8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]  #index for id - education - sex - age -PlaceOfDeathAndDecedentsStatus - MaritalStatus -MannerOfDeath - Race
# features = diseaseData[:,idx_IN_columns]
# featuresId = (shuffeledMat[:, 0])[:,None]
