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
from sklearn.neural_network import MLPClassifier


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

# print(features_label)
# for i in floatFeaturesMatrix:
#     print(i)


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

'''PCA'''
trainInp = np.array(PCA.PCA(floatFeaturesMatrix, 3))

'''NNet'''

X = trainInp
y = classes.flatten()
clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(2, 15), random_state=1)
clf.fit(X, y)                         


test_predicted = clf.predict( np.array(PCA.PCA(trainIn, 3)))
print(test_predicted)
test_real = trainOut.flatten()
print(test_real)

print('all')
ResultAnalyzer.confusionMatrix(test_real, test_predicted)
