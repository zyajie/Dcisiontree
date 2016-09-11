# Dcisiontree
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

allElectronicsData =open(r'D:\tree.csv')
reader = csv.reader(allElectronicsData)
headers = reader.next()

print(headers)

featureList =[]
labelList = []

for row in reader:
    labelList.append(row[len(row)-1])
    rowDict ={}
    for i in range(1,len(row)-1):
        rowDict[headers[i]]=row[i]
    featureList.append(rowDict)
print (featureList)

vec =DictVectorizer()
dumyX = vec.fit_transform(featureList).toarray()
print('dumyX:'+str(dumyX))
print(vec.get_feature_names)

lb = preprocessing.LabelBinarizer()
dumyY = lb.fit_transform(labelList)
print('dumyY:'+str(dumyY))


clf =tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(dumyX,dumyY)
print ('cf:'+str(clf))


with open('allElectronicInformationGainori.dot','w') as f:
    f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file = f)


