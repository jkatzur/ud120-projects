#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop("TOTAL")
features = ["from_messages", "to_messages"]
data = featureFormat(data_dict, features)

for point in data:
    from_messages = point[0]
    to_messages = point[1]
    matplotlib.pyplot.scatter( from_messages, to_messages )

matplotlib.pyplot.xlabel("from_messages")
matplotlib.pyplot.ylabel("to_messages")
matplotlib.pyplot.show()
