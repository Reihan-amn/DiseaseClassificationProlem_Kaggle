'''
Created on Nov 20, 2016

@author: Reihan
'''

import csv
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools


'''read data from file and convert it to numpy matrix'''
def getDataMatrixFromCSV(fileName):
    dataMatrix = np.array(list(csv.reader(open(fileName, "r+"), delimiter=',')))
    counter = 0
    print('type(dataMatrix ', type(dataMatrix), 'shape: ', dataMatrix.shape)
    smallerMatrix = []
    for row in dataMatrix:
        smallerMatrix.append(row)
        counter += 1
        if(counter == 100000):
            break    
    smallerMatrix = np.array(smallerMatrix) 
    print('type(smallerMatrix):', type(smallerMatrix), 'shape: ', smallerMatrix.shape)
    return smallerMatrix


'''shuffeling the data'''
def matrixShuffling(mat):
    matrix = np.random.permutation(mat)
    
    return matrix


'''converting the value type to float'''
def valueConvertToFloat(value):
    try:
        return(float(value))
    except:
        return value
    
def matrixConvertToFloat(matrix):
    floatMatrix = [[valueConvertToFloat(eachVal) for eachVal in row ] for row in matrix]   
    return  np.array(floatMatrix)


''' for all features do the normalization-column by column'''
def normalizingFeatures(data):
    normData = []
    for i in range(0, len(data[0])):
        feature = []
        feature = data[:, i]     
        mean = float(sum(feature) / float(len(feature)))
#         print('idex ' , i , 'sum: ' , sum(feature), ' lenght : ', len(feature), ' mean: ', mean)
        maxF = max(feature)
        minF = min(feature)
        normalizedVec = []
        
        for eachVal in feature: 
            if(maxF != minF): 
                newVal = float(eachVal - mean) / float(maxF - minF)
#                 print('type ', type(newVal), ' value ', newVal)
                normalizedVec.append(newVal)
            else:
                normalizedVec.append(0.0)    
        normData.append(normalizedVec)   
    return np.array(normData)


'''rotating the data matrix'''
def RowToColumnTranspositionMatrix(data):
    return data.T


'''data spliting to train - test - validation set'''
def dataSplitFactory(featuresMatrix, classVector, traintRatio, validationRatio, testRatio):  
    dataLen = len(featuresMatrix)
    testLen = int(testRatio*dataLen)+1      
    trainLen = int(traintRatio*dataLen)-1
    validationLen = dataLen - (trainLen + testLen)
    scope = trainLen + validationLen
    splitedFeatures = np.split(featuresMatrix, [testLen])
    splitedClass = np.split(classVector, [testLen])
#     splitedMatrix = np.split(featuresMatrix,[trainLen,scope])
    test_input = splitedFeatures[0]
    test_output = splitedClass[0]
    trainValidation_inputChunk = np.split(splitedFeatures[1],[trainLen])
    trainValidation_outputChunk = np.split(splitedClass[1],[trainLen])
    
    train_input =  trainValidation_inputChunk[0]
    train_output =  trainValidation_outputChunk[0]
    validation_input =  trainValidation_inputChunk[1]
    validation_output =  trainValidation_outputChunk[1]
    
    return train_input, train_output, validation_input, validation_output, test_input, test_output


'''extracting the label data -- specific column'''
def pickDiseasetargets(data, col_name):
    pattern = re.compile('[A-R]+')
    newMatrix = []
    with open('./categorizedData/im-balancedData.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(col_name)
        for i in range(0, data.shape[0]):
            j = 24;
            if(re.match(pattern, data[i][j])):
                newMatrix.append(data[i]) 
                writer.writerow(data[i])        
    return np.array(newMatrix)


'''
    for the class vector, we are changing the icd10 codes to integers
    the proper information related to this column is on the kaggle.com
    here we have 19 different classes
'''
def classDiscreteValueConverToDecimal(vector):
    vect = []

    for item in vector:
        if(item.startswith('A') or item.startswith('B')):
            vect.append(1.0)
        elif(item.startswith('C') or item.startswith('D0') or item.startswith('D1') or item.startswith('D2') or item.startswith('D3') or item.startswith('D4')):
            vect.append(2.0)
        elif(item.startswith('D5') or item.startswith('D6') or item.startswith('D7') or item.startswith('D8')):
            vect.append(3.0)
        elif(item.startswith('E')):
            vect.append(4.0)
        elif(item.startswith('F')):
            vect.append(5.0)
        elif(item.startswith('G')):
            vect.append(6.0)
        elif(item.startswith('H0') or item.startswith('H1') or item.startswith('H2') or item.startswith('H3') or item.startswith('H4') or item.startswith('H5')):
            vect.append(7.0)
        elif(item.startswith('H6') or item.startswith('H7') or item.startswith('H8') or item.startswith('H9')):
            vect.append(8.0)
        elif(item.startswith('I')):
            vect.append(9.0)
        elif(item.startswith('J')):
            vect.append(10.0)
        elif(item.startswith('K')):
            vect.append(11.0)  
        elif(item.startswith('L')):
            vect.append(12.0)
        elif(item.startswith('M')):
            vect.append(13.0)
        elif(item.startswith('N')):
            vect.append(14.0)
        elif(item.startswith('O')):
            vect.append(15.0)      
        elif(item.startswith('P')):
            vect.append(16.0)  
        elif(item.startswith('Q')):
            vect.append(17.0)  
        elif(item.startswith('R')):
            vect.append(18.0)
        else:
            vect.append(19.0)                  
                                                
    return np.array(vect)


'''converting discrete sex to number'''
def sexConv(value):
    '''
    female = 1
    male = 0
    '''
    if(value == 'F'):
        return 1
    else:
        return 0


'''converting discrete marital status to number'''
def maritalSConv(value):
    '''
    S  =  Never married, single  = 0
    M  =  Married                = 1
    W  =  Widowed                = 2
    D  =  Divorced               = 3
    U  =  Marital Status unknown = 4
    '''
    if(value == 'S'):
        return 1
    elif(value == 'M'):
        return 2
    elif(value == 'W'):
        return 3
    elif(value == 'D'):
        return 4
    else:
        return 5


'''converting discrete value to number'''
def InjuryAtWorkConv(value):
    if(value == 'Y'):
        return 0
    elif(value == 'N'):
        return 1
    else:
        return 2

'''converting discrete value to number'''
def MethodOfDispositionConv(value):
    '''
    B,Burial =0
    C,Cremation =1
    O,Other =2
    U,Unknown =3
     '''
    if(value == 'B'):
        return 0
    elif(value == 'C'):
        return 1
    elif(value == 'O'):
        return 2
    else:
        return 3


'''converting discrete value to number'''
def AutopsyConv(value):
    if(value == 'Y'):
        return 0
    elif(value == 'N'):
        return 1
    else:
        return 2


'''converting discrete value to number'''
def featureDiscreteToZeroOne(features, gender, marriage, InjuryAtWork, MethodOfDisposition, Autopsy):
    '''
    converting the discrete columns to decimal
    sex M/F to 0/1  
    Marital status to digit
    InjuryAtWork
    MethodOfDisposition
    Autopsy
'''
    newMatrix = []
    for i in range(0, len(features)):
        vec = []
        for j in range(0, len(features[i,:])):
            if j == gender:
                vec.append(sexConv(features[i,j]))
            elif(j == marriage):
                 vec.append(maritalSConv(features[i,j]))
            elif(j == InjuryAtWork):
                vec.append(InjuryAtWorkConv(features[i,j]))
            elif(j == MethodOfDisposition):
                vec.append(MethodOfDispositionConv(features[i,j]))  
            elif(j == Autopsy):
                vec.append(AutopsyConv(features[i,j]))        
            else:
                vec.append(features[i,j])
        newMatrix.append(vec)  
    return np.array(newMatrix)


'''balancing the data to have same number of instance for each class'''
def balancingData(imbalancedData, numOfInstances, columnsName):
    balancedMatrix = []
#     class1, class2, class3, class4, class5, class6, class7, class8, class9, class10, class11, class12, class13, class14, class15, class16, class17, class18    = []
    cl1 =0
    cl2 =0
    cl3 =0
    cl4 =0
    cl5 =0
    cl6 =0
    cl7 =0
    cl8 =0
    cl9 =0
    cl10 =0
    cl11 =0
    cl12 =0
    cl13 =0
    cl14 =0
    cl15 =0
    cl16 =0
    cl17 =0
    cl18 = 0
    with open('./categorizedData/balancedData.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columnsName)
        for i in range(0, len(imbalancedData)):
            print('i   ', i)
            j = 24;
            item = imbalancedData[i][j]
            if((cl1 <10) and (item.startswith('A') or item.startswith('B'))):
                print('class1  ' , imbalancedData[i],  '  cl1 ' , cl1)
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])
                cl1 +=1
            elif((cl2 <10) and (item.startswith('C') or item.startswith('D0') or item.startswith('D1') or item.startswith('D2') or item.startswith('D3') or item.startswith('D4'))):
                print('class2  ' , imbalancedData[i])
                balancedMatrix.append(imbalancedData[i])  
                writer.writerow(imbalancedData[i])     
                cl2 +=1  
            elif((cl3 <10) and (item.startswith('D5') or item.startswith('D6') or item.startswith('D7') or item.startswith('D8'))):
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])  
                cl3 +=1
            elif((cl4 <10) and item.startswith('E')):
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])  
                cl4 +=1
            elif((cl5 <10) and item.startswith('F')):
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])  
                cl5 +=1
            elif((cl6 <10) and item.startswith('G')):
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])  
                cl6 += 1
            elif((cl7 <10) and (item.startswith('H0') or item.startswith('H1') or item.startswith('H2') or item.startswith('H3') or item.startswith('H4') or item.startswith('H5'))):
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])  
                cl7 +=1
            elif((cl8 <10) and (item.startswith('H6') or item.startswith('H7') or item.startswith('H8') or item.startswith('H9'))):
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])  
                cl8 +=1
            elif((cl9 <10) and item.startswith('I')):
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])  
                cl9 +=1
            elif((cl10 <10) and item.startswith('J')):
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])  
                cl10 +=1
            elif((cl11 <10) and item.startswith('K')):
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])  
                cl11 +=1  
            elif((cl12 <10) and item.startswith('L')):
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])  
                cl12 +=1
            elif((cl13 <10) and  item.startswith('M')):
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])  
                cl13 +=1
            elif((cl14 <10) and  item.startswith('N')):
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])  
                cl14 +=1
            elif((cl15 <10) and  item.startswith('O')):
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])  
                cl15 +=1     
            elif((cl16 <10) and  item.startswith('P')):
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])  
                cl16 +=1  
            elif((cl17 <10) and  item.startswith('Q')):
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])  
                cl17 +=1  
            elif((cl18 <10) and  item.startswith('R')):
                balancedMatrix.append(imbalancedData[i])
                writer.writerow(imbalancedData[i])    
                cl18 +=1       
            elif (cl1 >=10 and cl2 >=10 and cl3 >=10 and cl4 >=10 and cl5 >=10 and cl6 >=10 and cl7 >=10 and cl8 >=10 and cl9 >=10 and cl10 >=10 and cl11 >= 10 and cl12>=10 and cl13 >=10 and cl14 >=10 and cl15 >= 10 and cl16 >=10 and cl17 >=10 and cl18 >=10 ):
                break
            else:
                continue
    return np.array(balancedMatrix)


def roundingNumbers(matrix):
    new_mat = []
    for row in matrix:
        vec= []
        for item in row:
          vec.append(round(item))
        new_mat.append(vec)
    return np.array(new_mat)


'''plotting the data'''
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')