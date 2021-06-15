import numpy as np
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


class KNNModel:
    
    def __init__(self,test_size):
        """
        c'est le constructeur de notre module,il contient les attribbuts 
        qu'on va utliser
        """
        self.letters = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
        self.data_path = "A_Z Handwritten Data.csv"
        self.test_size = test_size
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.y_train_cat = None
        self.y_test_cat = None
        self.history = None
        self.cm = None
        self.y_pred = None
        
    
    def getDataFrame(self):
        """
        cette methode return un objet Pandas qui contient  nos donnees.
        on a utliser astype(float) pour convertir les nombres en reels
        """
        return pd.read_csv(self.data_path).astype('float32')
    
    def getData(self):
        """
        cette methode return les donnees x (matrice des donnees)
        et y (les etiquettes).
        #selection des donnes.
        nous avons 785 variable.
        le premier variable est notre lettre (0:A,1:B,...,25:Z).
        le deuxieme variable est notre image (dim = (1,784) alors notre image en realite est de taille 28x28=784).
        on va donc stocker le premier variable dans y (target) est les autres dans X (data).
         .values pour rendre le type de x et y en numpy
        """
        df = self.getDataFrame()
        y = df.iloc[:,0].values
        X = df.iloc[:,1:].values 
        pca = PCA(n_components = 2)
        X = pca.fit_transform(X)
        print("information quantity transformed by each axe : " + 
              str(pca.explained_variance_ratio_))
        print("singular values : " + str(pca.singular_values_))
        print("dimension de X = " + str(X.shape))
        print("dimension de y = " + str(y.shape))
        return X,y
    
    def splitData(self):
        """
        on va diviser df en deux :
            -partie pour le trainning 
            -partie pour le test
        """
        X,y = self.getData()
        self.X_train, self.X_test,self.y_train, self.y_test = train_test_split(
            X, y, test_size = self.test_size)
        print("dimension de X_train = " + str(self.X_train.shape))
        print("dimension de y_train = " + str(self.y_train.shape))
        print("dimension de X_test = " + str(self.X_test.shape))
        print("dimension de y_test = " + str(self.y_test.shape))
        
        
    def buildModel(self):
        """
        cette methode sert a creer notre reseau de neurons:
            -Sequential():une liste qui contient les layers du reseau
            -Conv2D(filters=32, kernel_size=(3, 3)):il applique des filtres
            de taille 3*3 32fois (ces filtres changent a chaque iteration par adaboost)
            -MaxPool2D():pour garder que les pixel qui ont les plus grands niveau
            de gris (cette methode utlise pour reduire la dimension de notre image)
            -Flatten():pour rendre une image en une liste ((28,28) => (784,1))
            il concatent les lignes avec eux.
            -Dense():pour creer un layer
            -DropOut():pour elimuner des neurons dans la phase d'aprentissage 
            afin d'eviter l'overfiting
        """
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(self.X_train, self.y_train)
        
    
    
    
    def predict(self):
        """
        cette methode va predire les resultats des test afin de comparer ces predictions
        avec les vrai valeurs de y_test
        """
        self.y_pred = self.model.predict(self.X_test)
        print("dim de y_pred : " + str(self.y_pred.shape))
        print("dim de y_test : " + str(self.y_test.shape))
    
    def setConfusionMatrix(self):
        """
        cette methode return un confusion matrix qui va nous aider a evaluer 
        le model.
        """
        self.cm = confusion_matrix(self.y_test,self.y_pred)
        
        
    def getAccuracy(self):
        print("The accuracy  : "  + str(accuracy_score(self.y_pred,self.y_test())))

        
    
    
    
    