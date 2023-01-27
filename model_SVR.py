import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import matplotlib.pyplot as plt


# read the dataset and convert it into a dataframe
data = pd.read_csv('Salary_Data.csv')

# divide our dataset into attributes/features (X) and labels (y)
X = data['YearsExperience']
y = data['Salary']

# change the attribute's form
X = X[:,np.newaxis]

# build the model with C, gamma, and kernel parameters
model = SVR()
parameters = {
    'kernel': ['rbf'],
    'C':     [1000, 10000, 100000],
    'gamma': [0.5, 0.05,0.005]
}
grid_search = GridSearchCV(model, parameters)

# train the model
grid_search.fit(X,y)

print("SVR GridSearch score: "+str(grid_search.best_score_))
print("SVR GridSearch params: ")
print(grid_search.best_params_)

# build a new SVM model with the best parameter grid search results
new_model = SVR(C=100000, gamma=0.005, kernel='rbf')

# train the model
new_model.fit(X,y)

# create a plot
plt.scatter(X, y)
plt.plot(X, new_model.predict(X))
