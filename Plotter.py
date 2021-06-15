import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

class Plotter:
    def __init__(self):
        """
        c'est le constructeur de notre classe,il contient les attribbuts 
        qu'on va utliser
        """
        self.letters = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        
        
    def plotLettersCount(self):
        """
        cette methode permet d'afficher le nombre d'occureence de chaque lettre 
        dans notre base des donnees.
        
        """
        y_int = np.int64(self.y)
        count = np.zeros(26, dtype='int')
        for i in y_int:
            count[i] +=1
        
        alphabets = []
        for i in self.letters.values():
            alphabets.append(i)
        
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        ax.barh(alphabets, count)
        
        plt.xlabel("Nombre des elements")
        plt.ylabel("Lettres")
        plt.grid()
        plt.show()
        
        
    def plotImagesSample(self):
        """
        cette methode permet d'afficher 9 images aleatoire dans la parie trainning
        """
        shuff = shuffle(self.X_train[:100])

        fig, ax = plt.subplots(3,3, figsize = (10,10))
        axes = ax.flatten()
        
        for i in range(9):
            axes[i].imshow(shuff[i], cmap="gray")
        plt.show()
                
        
    def prdectedImages(self):
        """
        cette methode permet de predire les resultat ,afficher les images et aussi
        faire une comparaison des resultats
        """
        fig, axes = plt.subplots(5,5, figsize=(8,9))
        axes = axes.flatten()
        
        for i,ax in enumerate(axes):
            img = np.reshape(self.X_test[i], (28,28))
            ax.imshow(img,cmap='gray')
            
            pred = self.y_pred[i]
            ax.set_title("Pred:"+str(self.letters[pred]) + "-letter:"+str(self.letters[self.y_test[i]]))
            ax.set_yticklabels([])
            ax.set_xticklabels([])
    
        
        
        
        
        
        
        
        
        
        
        
        
        