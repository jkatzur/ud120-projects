#!/usr/bin/python

import sys
import pickle
import random
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import preprocessing
import numpy as np
import tester

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
###features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'perc_to', 'perc_from','perc_exer', 'perc_expense'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

### This is obviously a data error because it is just the sum of all the other values. When I missed this at first I wasted a TON of time.
data_dict.pop("TOTAL")
### This isn't a person so it cannot be arrested or convicted of a crime. I removed
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")
### I removed Ken Lay because he, essentially, got paid SO MUCH money that it destroyed all of the compensation numbers
data_dict.pop("LAY KENNETH L")
### Removing Mark Frevert is the most "controversial" outlier removal I did. Whenever I removed him the models improved enormously.
### I initially thought to remove him because he had such a high amount of "Other" comp, plus the fact that he was a very high paid
### closely connected exec who was apparently not convicted ever. Perhaps there's a nefarious thing here, but I'll that for the journalists
data_dict.pop("FREVERT MARK A")


### Task 3: Create new feature(s)
### his is creating some new features that I wanted to test.
for p in data_dict:
    ### This is a feature to figure out the % of messages a person received from person of interest compared to how many messaegs they got total.
    ### This normalizes against the sheer volume of this person's email.
	if (data_dict[p]['from_this_person_to_poi'] != 'NaN') & (data_dict[p]['to_messages'] != 'NaN'):
		data_dict[p]['perc_to'] = float(data_dict[p]['from_this_person_to_poi']) / float(data_dict[p]['to_messages'])
	else:
		data_dict[p]['perc_to'] = 'NaN'

    ### similar to above, this looks at the % of messages they sent to a poi
	if (data_dict[p]['from_poi_to_this_person'] != 'NaN') & (data_dict[p]['from_messages'] != 'NaN'):
		data_dict[p]['perc_from'] = float(data_dict[p]['from_poi_to_this_person']) / float(data_dict[p]['from_messages'])
	else:
		data_dict[p]['perc_from'] = 'NaN'

    ### This looks at what percentage of the person's stock options they exercised. It helps figure out if they believed
    ### the stock would continue to go up (and they wanted to hold it), or they wanted to cash out asap.
	if (data_dict[p]['exercised_stock_options'] != 'NaN') & (data_dict[p]['total_stock_value'] != 'NaN'):
		data_dict[p]['perc_exer'] = float(data_dict[p]['exercised_stock_options']) / float(data_dict[p]['total_stock_value'])
	else:
		data_dict[p]['perc_exer'] = 'NaN'

    ### this looked to see the percentage of this person's expense reimbursement compared to salary. The thinking was that
    ### someone who was abusing the expense policy may be more likely to do other shady things.
	if (data_dict[p]['expenses'] != 'NaN') & (data_dict[p]['salary'] != 'NaN'):
		data_dict[p]['perc_expense'] = float(data_dict[p]['expenses']) / float(data_dict[p]['salary'])
	else:
		data_dict[p]['perc_expense'] = 'NaN'

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

###this sets up feature scaling with a MinMaxScaler for all of the features
min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)
labels, features = targetFeatureSplit(data)

### I perfomed significant testing on k=3, k=4, k=5 features (see the Excels) to determine
### that k_best = 4 was the right number of features
k_best = 4
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2
selector = SelectKBest(chi2, k=k_best)
selector.fit(features,labels)

### Here are the relevant features!
### We found out they were total_payments, deferred_income, expenses, and perc_to
print selector.scores_
print selector.pvalues_
print selector.get_support()

features_list = ['poi'] + np.array(features_list)[selector.get_support()].tolist()

print features_list

###print data[0]

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Here I loaded all of the packages to test a bunch of different algorithms

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

"""
### This is the code I ran for the Grid Search CV to test many different algorithms
### and write the results of all of that testing to a CSV to look at later. You
### can see the result CSVs in the folder.

trials = 1000
results = ['k_best,trials,alg,param,accuracy,precision,recall,f1']


clf = GaussianNB()
dump_classifier_and_data(clf, my_dataset, features_list)
clf, dataset, feature_list = tester.load_classifier_and_data()
scores = tester.test_classifier(clf, dataset, feature_list, trials)
if scores:
	results.append('%s, %s,' %(k_best, trials) + 'bayes,,' + ','.join(map(str,scores)))

### this just runs through all of the options for Decision Trees
for split in (2, 3):
	for leaf in (1, 2, 3):
		clf = tree.DecisionTreeClassifier(min_samples_split = split, min_samples_leaf = leaf)
		dump_classifier_and_data(clf, my_dataset, features_list)
		clf, dataset, feature_list = tester.load_classifier_and_data()
		scores = tester.test_classifier(clf, dataset, feature_list, trials)
		if scores:
			results.append('%s, %s,' %(k_best, trials) + 'decision_tree,' + 'split:%s;leaf:%s,' %(split, leaf) + ','.join(map(str,scores)))

### Options for KNN
for alg in ("auto", "ball_tree", "kd_tree", "brute"):
	for n in (1, 2, 3, 4, 5):
		clf = KNeighborsClassifier(n_neighbors = n, algorithm = alg)
		dump_classifier_and_data(clf, my_dataset, features_list)
		clf, dataset, feature_list = tester.load_classifier_and_data()
		scores = tester.test_classifier(clf, dataset, feature_list, trials)
		if scores:
			results.append('%s, %s,' %(k_best, trials) + 'knn,' + '%s - %s,' %(alg, n) + ','.join(map(str,scores)))

### for AdaBoostClassifier
for n in (1, 5, 10, 20):
	clf = AdaBoostClassifier(n_estimators = n)
	dump_classifier_and_data(clf, my_dataset, features_list)
	clf, dataset, feature_list = tester.load_classifier_and_data()
	scores = tester.test_classifier(clf, dataset, feature_list, trials)
	if scores:
		results.append('%s, %s,' %(k_best, trials) + 'ada,' + '%s,' %(n) + ','.join(map(str,scores)))

### RandomForestClassifier
for n in (1, 5, 10, 25):
	clf = RandomForestClassifier(n_estimators = n)
	dump_classifier_and_data(clf, my_dataset, features_list)
	clf, dataset, feature_list = tester.load_classifier_and_data()
	scores = tester.test_classifier(clf, dataset, feature_list, trials)
	if scores:
		results.append('%s, %s,' %(k_best, trials) + 'forest,' + '%s,' %(n) + ','.join(map(str,scores)))

### SVC
for n in (1.0, 10.0, 100.0):
	clf = SVC(kernel = "rbf", C=n)
	dump_classifier_and_data(clf, my_dataset, features_list)
	clf, dataset, feature_list = tester.load_classifier_and_data()
	scores = tester.test_classifier(clf, dataset, feature_list, trials)
	if scores:
		results.append('%s, %s,' %(k_best, trials) + 'forest,' + '%s,' %(n) + ','.join(map(str,scores)))

### This prints the results so I couuld see them, but most importantly writes them to a CSV to review later
print results
csv_file = open("All_Data_Algs_k_%s" %(k_best) + "_trials_%s" %(trials) + ".csv", "w")
csv_file.write("\n".join(results))
csv_file.close()
"""
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### After exhaustive Grid Search CV work, here are the parameters that perform best and
### achieves the desired result described in Task 5
clf = tree.DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=1)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
print features_list
dump_classifier_and_data(clf, my_dataset, features_list)
