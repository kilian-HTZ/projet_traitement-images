#import des libs permetant l'interaction avec l'os et la génération de nombre aléatoire.
import os
import random



###########################
    #Initialisation
########################## 
    
#gestion des fichiers et des chemins
current_dir = "VOC2007/JPEGImages"
listedir=os.listdir(current_dir)


#cacule de compteur et du pourcentage de séparation

split_pct = 20;
nbr_im=len(listedir)
cpt_test= int((split_pct*nbr_im)/100)
cpt_train=nbr_im-cpt_test


#compteur permettant de
test=0
cpt_imtr=0
cpt_imte=0
     
while cpt_imtr+cpt_imte<nbr_im : #tant que le nombre de lignes dans les deux fichier txt n'est pas égales au nombre d'images on continue

###########################
    #test des doublons
########################## 
    
    if cpt_test>0 and cpt_train>0:
        #génération des nombres aléatoires 
        random_dir = random.randint(0, 1) #choix du dossier train ou test en passant le flag random_dir à 0 ou 1
        random_number = random.randint(1, nbr_im) #choix de l'image
        data=current_dir + "/" + str(random_number) + '.png' + "\n" #chemin de l'image choisi par random_number à inscrire dans le fichier txt
        
        
        #test pour éviter les doublons dans un fichier ou qu'une image se trouve dans test et train
        #pour test:
        with open("VOC2007/ImageSets/test.txt", "r") as file:#ouverture du fichier test 
    
            for line in file:#parcours des lignes
            
                if (data in line): #si on a déjà l'image dans test on passe le flag test à 1
                    test=1
                else: continue
            file.close() #on ferme le fichier pour le rouvrire en mode add 
        #pour train:
        with open("VOC2007/ImageSets/train.txt", "r") as File:
                for line in File:
            
                    if data in line:
                        test=1
                    else: continue
                file.close()
                
                
 ##############################             
     #Ecriture des fichiers
 ##############################
        if test==0 and random_dir==1:#si le flag random dir est à 1 on écrit dans test.txt
                file_val = open("VOC2007/ImageSets/test.txt", "a")#ouvre en mode add 
                file_val.write(data)
                cpt_test=cpt_test-1 #décrementation du compteur d'image test
                cpt_imte+=1 #incrément de la condition d'arrêt
        
        elif test==0:
            file_train = open("VOC2007/ImageSets/train.txt", "a") 
            file_train.write(data)
            cpt_train=cpt_train-1
            cpt_imtr+=1#incrément de la condition d'arrêt
        else: test=0 #sinon on repasse le flag test à 0 et on refait un tirage
    
    #Si on a déjà toutes les images de test alors il reste uniquement à affecter les images à train
    #on garde l'aléatoire pour brasser totalement les données
    #même fonctionnement qu'avant (CF ligne 28 à 68)
    elif cpt_train>0:
        random_number = random.randint(1, nbr_im) #choix de l'image
        data=current_dir + "/" + str(random_number) + '.png' + "\n"
        with open("VOC2007/ImageSets/train.txt", "r") as File:
                for line in File:
                    if data in line:
                        test=1
                    else: continue
                file.close()
        if test==0:
            file_train = open("VOC2007/ImageSets/train.txt", "a") 
            file_train.write(data)
            cpt_train=cpt_train-1
            cpt_imtr+=1
               
        else:test=0
          
        
    #fermeture des fichiers
file_train.close()
file_val.close()