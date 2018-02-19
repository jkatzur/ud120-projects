#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import math
    import numpy as np
    source_data = []
    
    for i in range(0,len(predictions)):
    	source_data.append((ages[i][0],net_worths[i][0],math.pow((net_worths[i] - predictions[i]),2)))

	clean_matrix = np.matrix(source_data,[('age','f'),('net_worth','f'),('error','f')])
	clean_matrix.sort(order='error')
	cleaned_data = clean_matrix.tolist()[0]
	cleaned_data = cleaned_data[:81]
    return cleaned_data
