#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
count = 0
nas = dict.fromkeys(enron_data["SKILLING JEFFREY K"].keys(), 0)
for p in enron_data:
	if enron_data[p]['poi'] == 1:
		count += 1
	for key in enron_data[p]:
		if enron_data[p][key] == "NaN":
			nas[key] += 1
print "%i Total People\n%i POIs" %(len(enron_data),count)
for n in nas:
	print "%s has %i NAs (%f %%)" %(n, nas[n], float(nas[n]) / 146)