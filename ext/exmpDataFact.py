import csv
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools


def getDataMatrixFromCSV(fileName):
    dataMatrix = np.array(list(csv.reader(open(fileName, "r+"), delimiter=',')))
    counter = 0
    pattern = re.compile('[A-B]+')
    print('type(dataMatrix ', type(dataMatrix), 'shape: ', dataMatrix.shape)
    newMatrix = []
    cl1 =0
    cl2 = 0
    
    i = 0
    with open('./balancedData.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(dataMatrix)):
            print('i   ', i)
            j = 24;
            item = dataMatrix[i][j]
            if(item.startswith('A') or item.startswith('B')):
                print('class1  ' , dataMatrix[i])
                newMatrix.append(dataMatrix[i])
                writer.writerow(dataMatrix[i])
                cl1 +=1
            elif((cl2 <10) and (item.startswith('C') or item.startswith('D0') or item.startswith('D1') or item.startswith('D2') or item.startswith('D3') or item.startswith('D4'))):
                print('class2  ' , dataMatrix[i])
                newMatrix.append(dataMatrix[i])
                writer.writerow(dataMatrix[i])       
                cl2 +=1       
            elif (cl1 >=10 and cl2 >=10):
                break
            else:
                continue
            
    return np.array(newMatrix)

def classDiscreteToInteger(vector):
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

def pickDiseasetargets(data):
    pattern = re.compile('[A-R]+')
    newMatrix = []
    print('Data: ', type(data), data.shape)
    print('data at index 257158:', data[257159][24])
    for i in range(0, data.shape[0]):
        j = 24;
        if(re.match(pattern, data[i][j])):
            newMatrix.append(data[i])         
    return np.array(newMatrix)