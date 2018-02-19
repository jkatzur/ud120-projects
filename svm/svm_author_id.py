#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
from sklearn.svm import SVC
###features_train = features_train[:len(features_train)/100] 
###labels_train = labels_train[:len(labels_train)/100] 

#c_options = [10000.0]
#for c in c_options:

clf = SVC(kernel = "rbf", C=10000.0)

#print "-----------------------------------------"
#print "I am using the %f C value" %(c)

t0 = time()
clf.fit(features_train, labels_train)
print "		training time:", round(time()-t0, 3), "s"
result = clf.predict(features_test)

count = 0
for test in result:
	if test == 1:
		count+=1

print "We predict that Chris wrote %f emails" %(count)

#t0 = time()
#score = clf.score(features_test, labels_test)
#print "		scoring time:", round(time()-t0, 3), "s"
#print "		score is %r" %(score)

#########################################################


