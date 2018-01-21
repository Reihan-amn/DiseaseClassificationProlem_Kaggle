from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics.scorer import accuracy_score

'''confusion matrix'''
def confusionMatrix(true_class, predicted_class):
    nf_mat = confusion_matrix(true_class, predicted_class)
    accuracy = accuracy_score(true_class, predicted_class)
    x = np.array(nf_mat)
    print('accuracy  ' , accuracy , '  con-mat: ',  precision_recall_fscore_support(true_class, predicted_class, average='macro'))
