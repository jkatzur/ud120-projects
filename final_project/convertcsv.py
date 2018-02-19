#!/usr/bin/python

""" 
I am a spreadsheet jockey in my job and very comfortable with data exploration in Excel
Obviously, Python is incredibly powerful for Machine Learning and unstructured
data analysis. That says, once we have the data in a tabular form (which is easy to do
from pickle), spreadsheet makes it simple to visually interact with your data quickly.
"""

import pickle
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
csv_source = ""
csv_source += ((','.join(enron_data["SKILLING JEFFREY K"].keys())) + "\n")
for p in enron_data:
	line = ""
	for key in enron_data[p]:
		line += str(enron_data[p][key])
		line += ","
	line += p
	csv_source += (line + "\n")

csv_file = open("EnronData.csv", "w")
csv_file.write(csv_source)
csv_file.close()