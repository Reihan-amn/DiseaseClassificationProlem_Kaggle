from ext import exmpDataFact

Bclass = exmpDataFact.getDataMatrixFromCSV("./DeathRecords.csv")
print('type: ', type(Bclass),' shape: ',Bclass.shape)

''''first row deleted '''
# columns_name = np.array(data2DMat[0])
# features_label =  np.delete(columns_name, [0,24], axis = None)
# data2DMat = np.delete(data2DMat, (0), axis=0) 
# 
# 
# '''class labelizing'''
# clss = data2DMat[:, 24]
# print("class lenght:" ,len(clss))
# clssa = exmpDataFact.classDiscreteToInteger(clss)
# classes = clssa[:,None]