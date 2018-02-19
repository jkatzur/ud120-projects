# Udacity Machine Learning Course
This repo contains all of my projects for the Udacity Machine Learning course.
This course covered the entire gamut from starting with raw, messy, messy data
you want to run models on all the way through building an accurate predictive model.
The final project for the course was to predict which executives at Enron were
eventually arrested and convicted of financial fraud / abuse.

To do this we used the Enron Email corpus, as well as other data sets we could
find on executives at Enron (such as their compensation, age, gender).

It was a fun and interesting way to learn Machine Learning as well as how to use
raw text data as input into a model.

## Course Review
The overall lesson from this course was how to go from...

Dataset --> Features --> Algorithm --> Evaluation
(and how this process is not linear, you optimize and move backwards and forwards)

### Dataset
When you begin with new data you need to determine:
* What am I interested in / what do I want to predict?
* Do I have enough data to test?
* Can I define the question?
* Can I generate enough features? <-- do I know enough about each input to generate a reasonable prediction
See /datasets_questions for example of how you start

### Features
Once I answer those questions positively I begin the feature process
* Exploration: Visualize the features, inspect for correlations, remove outliers, and clean the data. It is tremendously important to use this opportunity to remove outliers. They can have dramatic negative impact on your eventual results, and often this is just because of some data input error. Spending the time here up front can save you a ton of time later! See /evaluation and /outliers for examples.
* Creation: once you have played with the data you want to create the specific features you will evaluate.
* Representation: you then must turn these features into a numeric value that can be used as input to ML / AI algorithms. This means discretization and quantifying inputs like raw emails, which we do with vectorization. See /text_learning for example.
* Feature Scaling: Once we have our features we need to project them for maximum impact. The prototypical example is if we have height, we don't want the features to be listed in inches, where a 5' person (60 inches) doesn't seem that far, on a percentage basis, from an NBA player who is 6'10" (82 inches). We should scale to something like "inches above 5 feet tall". This would also apply to test scores.
* Feature Selection: In this step we choose the best features to use in our model. We are figuring out which of the plethora of inputs we may have at our disposal actually help predict the result. Common algos to help here are KBest, Percentile, and Recursive Feature selection. See /feature_select for more.
* Transforms: Lastly, we sometimes need to transform our feature inputs in to features that are more complex, but more explanative. For example, when predicting home value there is no quantitative value that defines a "Neighborhood", but if you have data on school rank, walkability, crime, and food options, those may all combine in to a "Neighborhood" score. To test this we use Principal Component Analysis, which can transform source features in to the principal components that really predict the results. See /pca for more.

### Algorithm
Once I have selected my feature I need to pick the best algorithm to optimize my prediction.
First I pick my algorithm type --> labels or no labels?
* NO Labels: If no labels I must pick an unsupervised algorithm like KMeans Clustering, PCA, or Spectral Clustering. These algos tell us which points are similar to each other without a label definition. See /pca and /k_means
* YES Labels: in this case I need to determine if my data is ordered or not.
Is data ordered?
* NO data is not ordered: then I should use a Decision Tree, Naive Bayes, SVM, K Nearest, Logistic Regression, or some ensemble of those methods. See /svm, /naive_bayes, /decision_tree
* YES data is ordered: then I should use Linear Regression, SV Regression, Lasso Regression, or Decision Tree Regression.

Once I have determine which algorithm, I want to use I must tune it. Each of the algorithms have a variety of ways to tune them. This starts with visual inspection, but quickly should turn to Grid Search CV which creates a grid of all of the various inputs and determines which is the optimal model. See /final_project for a long example of this.

### Evaluation
Once you have picked a model you have to validate it. This is, in reality, part of the Algorithm step. You go back and forth constantly between aiming to pick an algorith, and evaluating it. Common methods here are to Validate via Test / Training split, Kfold, Visualize Results, and Pick Metrics like SSE / R^2, precision, recall, F1, ROC and optimize for this. See /evaluation.

### Final Project
In /final_project you will find an entire end-to-end machine learning process with the Enron data set. Some files to call out are:
* convertcsv.py: In this file I wrote a simple Python script to convert the pickle data in to a CSV so I could explore the data in Excel. This made the visualize step significantly easier for me. You can see that file at EnronData.csv
* poi_id.py: This is the most important file in the entire repo. It is the end to end process for the Enron project. I start with the raw data, remove outliers, build features, scale and transform them, pick my algo (via a long Grid Search CV process), and then run my final output
* All_Data_Algs_k5_trials_*: these are the .csvs I built to review the output of my Grid Search CV. This was very helpful in keeping track of all my testing and making sure I picked the best algorithm in the end.

This course was a lot of fun. I learned a lot and would recommend it to anyone. For more information go to: https://www.udacity.com/course/intro-to-machine-learning--ud120
