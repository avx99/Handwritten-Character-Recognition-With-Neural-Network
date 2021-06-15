from Model import *
from Plotter import *

if __name__ == "__main__":
    
######################################### inisialisation du model et de plotter   
    model = Model(test_size=0.25,learning_rate = 0.001,epochs = 10,batch_size = 50)   
    plotter = Plotter()
    # lecture des donnees
    df = model.getDataFrame()
    
    #affichage des 5 premiers lignes
    df.head(5)

    #description des donnees
    df.describe()
######################################### extraction des donnees et des etiquttes

    # X = matrice des donnees et y = target
    X,y = model.getData()
    
######################################### divisionn des donnees en deux groupes(test&train)

    model.splitData()
    
    #affichage des occurences de chaque lettre dans la base des donnees
    plotter.y = y
    plotter.plotLettersCount()    
    
######################################### affichage des images 

    #on va convertir les dimension des image pour les afficher
    model.reshapeForPlotting()

    #affichage des images de notre base des donnees
    plotter.X_train = model.X_train
    plotter.plotImagesSample()




######################################### manipulation des donnees

    #on va convertir les dimension des image pour les passer dans le CNN
    model.reshapeForCNN()
    
    #puisqu'on va utiliser categorical_crossentropy comme un losss function
    #nous devons convertir y_test et _train en categories
    model.numToCategorical()
    
    #maintenant on va creer notre modele
    model.buildModel()

    #on va compiler le modele avec Adam optimizer(qui ce base sur le stochastic 
    #gredient descente) avec un learning rate de 0.001 (learning rate = le pas
    # avec lequel on change Pk par Pk+1)
    model.compileTheModel()    

    #on passe maintenant a la phase d'apprentissage avec un batch size de 50
    #image et avec 10 itération pour la modification des weights Wi
    #batch_size = 50 signifie que apres le passage de 50 images on va modifier les Wi
    model.getLearningHistory()

    #maintenant on va afficher un sommaire de notre model(a chaque etape
    #on dimunie la taille de l'image)
    print(model.getTheModelSummary())
    
    #apres l'apprentissage il est recommander de stocker le model
    #dans un fichier pour gagner le temps d'apprentissage
    model.setH5File()

############################################# evaluation de notre model & affichage 
#                                             des resultats                                          
    
    #on va creer la vecteur de prediction y_pred
    model.predict()
    
    #affichage des resultats
    plotter.y_pred = model.y_pred
    plotter.y_test = model.y_test
    plotter.X_test = model.X_test
    plotter.prdectedImages()
    
    #on va comparer y_pred avec y_test en utilisant confusion matrix
    model.setConfusionMatrix()
    
    #nous remarqouns que a  chaque ligne de cm,l'element en diagonale est le plus
    #grans => nous avons presque une bonne resultat
    cm = model.cm
    


















