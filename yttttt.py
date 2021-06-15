import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


df = pd.read_csv('A_Z Handwritten Data.csv')

y = df.iloc[:,0].values
X = df.iloc[:,1:].values 
pca = PCA(n_components = 2)
X = pca.fit_transform(X)
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.25)

letters = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

model = KNeighborsClassifier(n_neighbors=26)

model.fit(X_train,y_train)


y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
