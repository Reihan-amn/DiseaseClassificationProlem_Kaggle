import DataIOFactory
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import FeatureSelection_Chi2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsRegressor
import ResultAnalyzer
import PCA


'''reading balanced data from csv file'''
balancedData = DataIOFactory.getDataMatrixFromCSV("./categorizedData/balancedData.csv")

''' extracting feature labels and class label'''
columns_name = np.array(balancedData[0])#first rows has the column names
features_label =  np.delete(columns_name, [0,24], axis = None) #removing the id and class column
balancedData = np.delete(balancedData, (0), axis=0) #removing the label row from data 


'''shuffling the data'''
balancedData = DataIOFactory.matrixShuffling(balancedData)


'''Column 24, icd10 is our class and the rest are features '''
clss = balancedData[:, 24]
# clss = np.delete(clss, (0), axis=0) #removing the label from this list - first column
print("class lenght:" ,len(clss))
classes = DataIOFactory.classDiscreteValueConverToDecimal(clss)[:,None] #clss is just 1D array, we have to convert it to 2D array
print('class shape' , classes.shape)


'''select the columns as input features - all columns but 0 and 24'''
features =  np.delete(balancedData, [0,24], axis = 1)


'''converting descrete featuers to digit'''
featureMatrix = DataIOFactory.featureDiscreteToZeroOne(features, 5, 14, 17, 19, 20)  


'''converting all features to float64'''
floatFeaturesMatrix = DataIOFactory.matrixConvertToFloat(featureMatrix)
print('size:', floatFeaturesMatrix.shape)


'''normalizing the data'''
inputNormalizedMatrix = DataIOFactory.RowToColumnTranspositionMatrix(DataIOFactory.normalizingFeatures(floatFeaturesMatrix))
# print(inputNormalizedMatrix)


'''spliting the data into test and train and validation set based on different portions'''
trainIn, trainOut, validationIn, validationOut, testIn, testOut = DataIOFactory.dataSplitFactory(floatFeaturesMatrix, classes, 0.8, 0.1, 0.1)
# print('trainIn' ,trainIn.shape)
# print('trainout', trainOut.shape)
# print('feat', floatFeaturesMatrix.shape)
# print(classes)
# print('class', classes.shape)



'''chi2 feature selection'''
sorted_features_score = FeatureSelection_Chi2.Chi2_featureSelection(floatFeaturesMatrix, classes, features_label, 'all') 



n_neighbors = 3
'''classification based on all features'''
neigh = KNeighborsRegressor(n_neighbors)
neigh.fit(trainIn, trainOut)
valid_pred =  neigh.predict(floatFeaturesMatrix)
valid_pred = DataIOFactory.roundingNumbers(valid_pred)
# for i in range(0, len(valid_pred)):
#     print('predict: ' , valid_pred[i], '  real: ' , classes[i] )
n_real = classes.flatten()
n_predict = valid_pred.flatten()

'''results'''
print('all')
ResultAnalyzer.confusionMatrix(n_real, n_predict)
'''------------------------------------------------------------'''
'''classification based on top 10 features'''
'''extracting the sub-matrix of top 10 features from the original matrix'''
top10 = sorted_features_score[:10]
top10Index = []
for i in range(10):
    top10Index.append(int(top10[i][0]))
trainIn_10 = trainIn[:,top10Index]
trainOut_10 = trainOut
floatFeaturesMatrix_10 = floatFeaturesMatrix[:, top10Index]

neigh = KNeighborsRegressor(n_neighbors)
neigh.fit(trainIn_10, trainOut_10)
valid_pred =  neigh.predict(floatFeaturesMatrix_10)
valid_pred = DataIOFactory.roundingNumbers(valid_pred)
# for i in range(0, len(valid_pred)):
#     print('predict: ' , valid_pred[i], '  real: ' , classes[i] )
n_real = classes.flatten()
n_predict = valid_pred.flatten()
 
'''results'''
print("best 10")
ResultAnalyzer.confusionMatrix(n_real, n_predict)
'''----------------------------------------------------------'''
'''classification based on top 5 features'''
'''extracting the sub-matrix of top 5 features from the original matrix'''
top5 = sorted_features_score[:5]
top5Index = []
for i in range(5):
    top5Index.append(int(top5[i][0]))
trainIn_5 = trainIn[:,top5Index]
trainOut_5 = trainOut
floatFeaturesMatrix_5 = floatFeaturesMatrix[:, top5Index]

neigh = KNeighborsRegressor(n_neighbors)
neigh.fit(trainIn_5, trainOut_5)
valid_pred =  neigh.predict(floatFeaturesMatrix_5)
valid_pred = DataIOFactory.roundingNumbers(valid_pred)
# for i in range(0, len(valid_pred)):
#     print('predict: ' , valid_pred[i], '  real: ' , classes[i] )
n_real = classes.flatten()
n_predict = valid_pred.flatten()
 
'''results'''
print("best 5")
ResultAnalyzer.confusionMatrix(n_real, n_predict)
    
# # Create color maps
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
# n_neighbors = 18
# h = .02  # step size in the mesh
# for weights in ['uniform', 'distance']:
#     # we create an instance of Neighbours Classifier and fit the data.
#     clf = neighbors.KNeighborsClassifier(n_neighbors , weights=weights)
#     clf.fit(trainIn, trainOut)
#     X = trainIn
#     y = trainOut
# 
#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, x_max]x[y_min, y_max].
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# 
#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.figure()
#     plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# 
#     # Plot also the training points
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.title("3-Class classification (k = %i, weights = '%s')"
#               % (n_neighbors, weights))
# 
# plt.show()
'''classification after applying PCA'''
'''classification after applying PCA'''
trainInp = np.array(PCA.PCA(floatFeaturesMatrix, 3))
neigh = KNeighborsRegressor(3)
neigh.fit(trainInp, classes)
valid_pred =  neigh.predict(trainInp)
valid_pred_1 = DataIOFactory.roundingNumbers(valid_pred)
 
n_real_1 = classes.flatten()
n_predict_1 = valid_pred_1.flatten()
print('real ' , n_real_1.shape)
print('predict ', n_predict_1.shape)
  
'''results'''
print('PCA')
ResultAnalyzer.confusionMatrix(n_real_1, n_predict_1)


# neigh = KNeighborsRegressor(3)
# neigh.fit(trainInp, trainOut)
# valid_pred =  neigh.predict(floatFeaturesMatrix)
# valid_pred_1 = DataIOFactory.roundingNumbers(valid_pred)
# # for i in range(0, len(valid_pred)):
# #     print('predict: ' , valid_pred[i], '  real: ' , classes[i] )
# n_real_1 = classes.flatten()
# n_predict_1 = valid_pred_1.flatten()
# 
# '''results'''
# print('PCA')
# ResultAnalyzer.confusionMatrix(n_real_1, n_predict_1)


