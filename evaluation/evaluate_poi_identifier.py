#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### Split into train and test
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)   

### Build initial prediction
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

### Evaluation Metrics
predicted = clf.predict(features_test)
poi_correct = 0
not_poi_correct = 0
false_positive = 0
false_negative = 0
for predict, actual in zip(predicted, labels_test):
	if (predict == 1.0) & (actual == 1.0):
		poi_correct += 1
	elif (predict == 0.0) & (actual == 0.0):
		not_poi_correct += 1
	elif (predict == 1.0) & (actual == 0):
		false_positive += 1
	elif (predict == 0) & (actual == 1):
		false_negative += 1

print "%i POI Correct\n%i NOT POI Correct\n%i False Positive\n%i False Negative" %(poi_correct, not_poi_correct, false_positive, false_negative)
print clf.score(features_test, labels_test)
print "Recall: %f Precision: %f" %(float(poi_correct) / (poi_correct + false_negative), float(poi_correct) / (poi_correct + false_positive))
print "Recall: %f Precision: %f" %(recall_score(labels_test, predicted), precision_score(labels_test, predicted))
