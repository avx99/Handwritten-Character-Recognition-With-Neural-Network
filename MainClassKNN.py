from KNNModel import *




if __name__ == "__main__":
    
######################################### inisialisation du model et de plotter   
    model = KNNModel(test_size=0.25)   
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



######################################### manipulation des donnees

    
    #maintenant on va creer notre modele
    model.buildModel()

############################################# evaluation de notre model & affichage 
#                                             des resultats                                          
    
    #on va creer la vecteur de prediction y_pred
    model.predict()
    
    
    #on va comparer y_pred avec y_test en utilisant confusion matrix
    model.setConfusionMatrix()
    
    #nous remarqouns que a  chaque ligne de cm,l'element en diagonale est le plus
    #grands => nous avons presque une bonne resultat
    cm = model.cm
    


















