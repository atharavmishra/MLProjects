import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#IMPORTING THE DATASET
dataset=pd.read_csv('Iris.csv')
x=dataset.iloc[:,1:5]
y=dataset.iloc[:,[5]]
#removing useless data
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures=SelectKBest(score_func=chi2,k=4)
bestfeatures=bestfeatures.fit(x,y)
score_=bestfeatures.scores_ #this is done just to visualise the score of our bestfeatures
dfscores=pd.DataFrame(score_)
dfcoloumns=pd.DataFrame(x.columns)
featurescores=pd.concat([dfcoloumns, dfscores], axis=1)
featurescores.columns=['feautres_flower','score']
print(featurescores.nlargest(4,'score'))

#finding correlation between features
import seaborn as sns
dataset.corr()
sns.jointplot(x='PetalWidthCm', y='PetalLengthCm', data=dataset, kind='reg')
sns.heatmap(dataset.corr())#it is the most important graph it defines the relation between features using densities of colors
sns.pairplot(dataset)#it uses permutations and combinations and plots the graph of two featurs at a time
sns.pairplot(dataset,hue='PetalLengthCm')#here the correlation between petallength and every other feature will be plotted
sns.countplot('Species',data=dataset)
sns.boxplot('PetalLengthCm','PetalWidthCm',data=dataset)

x=dataset.iloc[:,3:5].values
y=dataset.iloc[:,5].values
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x=LabelEncoder()
y[:, 0]=labelencoder_x.fit_transform(y[:, 0])
onehotencoder = ColumnTransformer([("Species", OneHotEncoder(), [0])],    remainder = 'passthrough')
y=onehotencoder.fit_transform(y)'''
#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.20, random_state=0)

#fitting the logistic regression into the training set
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_train, y_train)

#cross validation score
from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,x,y,cv=5)
print(score.mean())

#predictig the data
y_pred=classifier.predict(x_test)

#testing the accuracy
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
classifier.score(x_test,y_test)
print('training_accuracy:',classifier.score(x_train,y_train))
print('test_accuracy:',classifier.score(x_test,y_test))

'''from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()''' 