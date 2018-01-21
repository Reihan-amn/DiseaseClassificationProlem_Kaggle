from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest 
import numpy as np


'''chi2 feature selection'''
def Chi2_featureSelection(trainIn, trainOut, features_label, numOfFeaturesToBeConsidered):
    print("\n-----------------------------------------")
    print("Implementing Chi2 feature selection....")
    X_new = SelectKBest(chi2, k = numOfFeaturesToBeConsidered).fit(trainIn, trainOut)
    print('scores')
    print(type(X_new.scores_))
    print('feature size' , len(features_label))
    print('score size' , len(X_new.scores_))

    feat_score = []
    for i in range(0, len(features_label)):
        temp = []
        # print('features' , features_label[i], '        score: '  ,X_new.scores_[i] )
        temp.append(i)
        temp.append(features_label[i])
        temp.append(X_new.scores_[i])
#         print(temp)
        feat_score.append(temp)

    '''sorting the features based on their score'''
    sortedList = sorted(feat_score, key=(lambda x: x[1]), reverse = True)
    # print(sortedList)
    print("Done with Chi2 feature selection....")
    return np.array(sortedList)