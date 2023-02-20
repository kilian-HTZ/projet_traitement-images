# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 23:53:18 2023

@author: HEITZ Kilian
"""

#import des libs os pour diaoguer avec le système d'exploitation et shutil pour la focntion shutil.copy
import os
import shutil

path = "train_data_pascal" # Dossier contenant les fichiers XML et JPG

cpt=0 #permet de donner des numéros aux fichier

for filename in os.listdir(path): #Parcours de la liste contenant les fichiers pas les ELTS


    if filename.endswith(".jpg"):#si le fichier est un point 'jpf' alors on cherche sont .xml et les renommes 
        
        jpg_file = os.path.splitext(filename)[0]
        xml_file = jpg_file + ".xml"

        if os.path.exists("train_data_pascal/"+xml_file): #test pour s'assurer qu'il exist bien '.xml' avec le même nom
        
            #création des nouveaux chemin
            new_jpg="train_data_pascal/"+str(cpt)+'.jpg'
            new_xml="train_data_pascal/"+str(cpt)+'.xml'
            
            #copy des fichiers en cpt.jpg et cpt.xml
            shutil.copy("train_data_pascal/"+filename, new_jpg)
            shutil.copy("train_data_pascal/"+xml_file, new_xml)
            
            #suppression des anciens fichier
            os.remove("train_data_pascal/"+filename)
            os.remove("train_data_pascal/"+xml_file)
        cpt+=1