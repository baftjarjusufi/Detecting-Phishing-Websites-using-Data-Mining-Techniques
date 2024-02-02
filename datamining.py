import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Objective: Is to detect phishing websites to prevent "phishing"

#Loading the data from csv file to be stored in a dataframe.

dataset = pd.read_csv('dataset.csv')
dataset.head()
#print("",dataset.head())



"""""
Familiarizing with Data
In this step, few dataframe methods are used to look into the data and its features.
"""
#Checking the shape of the dataset
dataset.shape
print("Dataset shape:",dataset.shape,"\n")

#Listing the features of the dataset
dataset.columns
#print("Dataset Columns",dataset.columns,) print if needed !

#Information about the dataset
dataset.info()
print("\n")
"""""
Visualizing the data:
Few plots and graphs are displayed to find how the data is distributed and the how features are related to each other.
"""




#Plotting the data distribution

dataset.hist(bins = 50,figsize = (20  ,8))
plt.title('Data Distribution plot:')
plt.show() #Showing the plot

#Correlation heatmap

plt.figure(figsize=(15,13))
plt.title('Correlation HeatMap:')
sns.heatmap(dataset.corr(), cmap=sns.diverging_palette(20, 0, n=200), vmin=-1, vmax=1, center=0,)
plt.show()


"""
K-Means clustering algorithm
"""
#standardize data
data1 = dataset
data1

# Building kmeans model
k = 3   #Category of clustering
iteration = 500  #Maximum cycles of clustering
data_zs = 1.0*(data1-data1.mean())/data1.std()   #Data standardization
data_zs
from sklearn.cluster import KMeans
model = KMeans(n_clusters = k, max_iter = iteration, random_state = 1234) #It is classified as k and the number of concurrent is 4
model.fit(data_zs)

# Result display
#Simple print results
r1 = pd.Series(model.labels_).value_counts()  #Count the number of each category
r2 = pd.DataFrame(model.cluster_centers_)  #Find the cluster center
r = pd.concat([r2,r1],axis =1)  #Get the number under the category corresponding to the cluster center

r.columns = list(data1.columns) + ['Number of clusters']  #header
print("K mean model:",r)

r = pd.concat([data1, pd.Series(model.labels_,index =data1.index)],axis =1)
r.columns = list(data1.columns) + ['Clustering category']

#Here, we clean the data by applying data preprocesssing techniques and transform the data to use it in the models.

dataset.describe()
print("Description of the Dataset:",dataset.describe())
#The above obtained result shows that the most of the data is made of 0's & 1's except 'index'.
#Dropping the Index column because it is not relavent for the models we need.
data = dataset.drop(['index'], axis = 1).copy()

#checking the data for null or missing values from the data for fixes
data.isnull().sum()

# shuffling the rows in the dataset so that when splitting the train and test set are equally distributed.
data = data.sample(frac=1).reset_index(drop=True)
data.head()





# Separating and assigned features and target columns to X & y.
y = data['Result']
X = data.drop('Result', axis=1)
X.shape, y.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled[:3]

# Splitting the dataset into train set and test sets, wih an 80-20 split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2, random_state = 12)
X_train.shape, X_test.shape

print("X,traindataSamples:",X_train.shape)
print("X,testdataSamples:",X_test.shape)

#importing packages for what we need for the upcoming models.
from sklearn.metrics import accuracy_score

# Creating holders to store the model performance results

ML_Model = []
acc_train = []
acc_test = []

#function to call for storing the results
def storeResults(model, a,b):
  ML_Model.append(model)
  acc_train.append(round(a, 3))
  acc_test.append(round(b, 3))

# Decision Tree model
from sklearn.tree import DecisionTreeClassifier

# instantiate the model
tree = DecisionTreeClassifier(max_depth = 5)
# fit the model
tree.fit(X_train, y_train)

#predicting the target value from the model for the samples we made
y_test_tree = tree.predict(X_test)
y_train_tree = tree.predict(X_train)

#Performance Evaluation:

#computing the accuracy of the model performance
acc_train_tree = accuracy_score(y_train,y_train_tree)
acc_test_tree = accuracy_score(y_test,y_test_tree)

print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree) ,"\n")
print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree),"\n")

#checking the feature importance in the model
plt.figure(figsize=(10,8))
n_features = X_train.shape[1]
plt.title('Decision Tree:')
plt.barh(range(n_features), tree.feature_importances_, align='center' , color = 'maroon')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()

#Storing the results:
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Decision Tree', acc_train_tree, acc_test_tree)

#Random Forest Classifier
# Random Forest model
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth=5)

# fit the model
forest.fit(X_train, y_train)

#predicting the target value from the model for the samples
y_test_forest = forest.predict(X_test)
y_train_forest = forest.predict(X_train)

#Performance Evaluation

#computing the accuracy of the model performance
acc_train_forest = accuracy_score(y_train,y_train_forest)
acc_test_forest = accuracy_score(y_test,y_test_forest)

print("Random forest: Accuracy on training Data: {:.3f}".format(acc_train_forest),"\n")
print("Random forest: Accuracy on test Data: {:.3f}".format(acc_test_forest),"\n")

#checking the feature improtance in the forest model

plt.figure(figsize=(9,7))
n_features = X_train.shape[1]
plt.title('Random Forest:')
plt.barh(range(n_features), forest.feature_importances_, align='center' , color = 'tomato')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()

#Storing the results:
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Random Forest', acc_train_forest, acc_test_forest)


#Multilayer Perceptrons (MLPs): Deep Learning
# Multilayer Perceptrons model
from sklearn.neural_network import MLPClassifier

# instantiate the model
mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=([100,100,100]))

# fit the model
mlp.fit(X_train, y_train)

#predicting the target value from the model for the samples
y_test_mlp = mlp.predict(X_test)
y_train_mlp = mlp.predict(X_train)

#Performance Evaluation:
#computing the accuracy of the model performance
acc_train_mlp = accuracy_score(y_train,y_train_mlp)
acc_test_mlp = accuracy_score(y_test,y_test_mlp)

print("Multilayer Perceptrons: Accuracy on training Data: {:.3f}".format(acc_train_mlp),"\n")
print("Multilayer Perceptrons: Accuracy on test Data: {:.3f}".format(acc_test_mlp),"\n")

#Storing the results:
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Multilayer Perceptrons', acc_train_mlp, acc_test_mlp)

from sklearn.ensemble import BaggingClassifier
bag_model = BaggingClassifier(
base_estimator=DecisionTreeClassifier(),
n_estimators=100,
max_samples=0.8,
bootstrap=True,
oob_score=True,
random_state=0
)
#Performance Evaluation:
#computing the accuracy of the model performance
bag_model.fit(X_train, y_train)
bag_model.oob_score_
bag_model.score(X_test, y_test)

acc_train_bag = bag_model.oob_score_
acc_test_bag = bag_model.score(X_test,y_test)

print("Bagging Classifier: Accuracy on training Data:",acc_train_bag,"\n")
print("Bagging Classifier: Accuracy on test Data:",acc_test_bag,"\n")

storeResults('Bagging Classifier', acc_train_bag, acc_test_bag)

"""""
Comparision of Models
To compare the models performance, a dataframe is created. 
The columns of this dataframe are the lists created to store the results of the model.
"""

#creating dataframe
results = pd.DataFrame({ 'ML Model': ML_Model,
    'Train Accuracy': acc_train,
    'Test Accuracy': acc_test})
results

#print("Comparision of Models Sorted:\n" ,results , "\n","\n")

#Sorting the datafram on accuracy

results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False)
print("Comparision of Models Sorted:\n")
print(results,"\n")

