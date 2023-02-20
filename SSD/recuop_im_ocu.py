# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 00:06:24 2023

@author: Legion5
"""

import xml.etree.ElementTree as ET

def count_Im_occu(nomtest,nomtrain,choix='test'):
    """
    

    Parameters
    ----------
    nomtest : string
        template avec le nom des images de la base de test on rentre le path
    nomtrain : string
        template avec le nom des images de la base d'entrainement' on rentre le path
    test_ot_train: permet de savoir le template choisi test ou train choix possible 'test'/'train'
    Returns un tuplet avec pour chaque classe (le nombre d'image du template ou elle apparait,le nombre d'occurence dans le tamplate)
    -------
    None.

    """



    #compteuroccurence totale
    
    cpt_amphiby_tot=0
    cpt_barge_tot=0
    cpt_carrier_tot=0
    cpt_cruiser_tot=0
    cpt_destroyer_tot=0
    cpt_fregate_tot=0
    cpt_lcs_tot=0
    cpt_patrol_tot=0
    cpt_submarine_tot=0
    cpt_supply_tot=0
    cpt_tender_tot=0
    cpt_pha_tot=0
    
    #compteur nombre d'image par classe
    cptI_amphiby=0
    cptI_barge=0
    cptI_carrier=0
    cptI_cruiser=0
    cptI_destroyer=0
    cptI_fregate=0
    cptI_lcs=0
    cptI_patrol=0
    cptI_submarine=0
    cptI_supply=0
    cptI_tender=0
    cptI_pha=0
    
    #ecompteur par image:
    #compteuroccurence totale
    cpt_amphiby=0
    cpt_barge=0
    cpt_carrier=0
    cpt_cruiser=0
    cpt_destroyer=0
    cpt_fregate=0
    cpt_lcs=0
    cpt_patrol=0
    cpt_submarine=0
    cpt_supply=0
    cpt_tender=0
    cpt_pha=0
    
    test=open(nomtest+".txt", "r")
    train=open(nomtrain+".txt", "r")
    
    L_train=[]
    L_test=[]
    
    
    
    if choix=='test':
        for elt in test:
            L_test.append(elt.split('.')[0].split('/')[-1])
      
            
        for elt in L_test:   
            tree = ET.parse("VOC2007/Annotations/"+str(elt)+'.xml')
            root = tree.getroot()
            
            
            for object_elem in root.iter('object'):
                name = object_elem.find('name').text
                
                if name=='carrier':
                    cpt_carrier+=1
                    
                elif name=='cruiser':
                    cpt_cruiser+=1
                elif name=='tender':
                    cpt_tender+=1
                elif name=='amphiby':
                    cpt_amphiby+=1
                elif name=='patrol':
                    cpt_patrol+=1
                elif name=='destroyer':
                    cpt_destroyer+=1
                elif name=='littoral combat ship':
                    cpt_lcs+=1
                elif name=='submarine':
                    cpt_submarine+=1
                elif name=='supply':
                    cpt_supply+=1
                elif name=='fregate':
                    cpt_fregate+=1
                elif name=='barge':
                    cpt_barge+=1
                elif name=='pha':
                    cpt_pha+=1
        
            
            if cpt_carrier!=0:
                cptI_carrier+=1
            if cpt_cruiser!=0:
                cptI_cruiser+=1
            if cpt_tender!=0:
                cptI_tender+=1
            if cpt_amphiby!=0:
                cptI_amphiby+=1
            if cpt_patrol!=0:
                cptI_patrol+=1
            if cpt_destroyer!=0:
                cptI_destroyer+=1
            if cpt_lcs!=0:
                cptI_lcs+=1
            if cpt_submarine!=0:
                cptI_submarine+=1
            if cpt_supply!=0:
                cptI_supply+=1
            if cpt_fregate!=0:
                cptI_fregate+=1
            if cpt_barge!=0:
                cptI_barge+=1
            if cpt_pha!=0:
                cptI_pha+=1
        
        
            #increment compteur images
            cpt_amphiby_tot=cpt_amphiby_tot+cpt_amphiby
            cpt_barge_tot=cpt_barge_tot+cpt_barge
            cpt_carrier_tot=cpt_carrier_tot+cpt_carrier
            cpt_cruiser_tot=cpt_cruiser_tot+cpt_cruiser
            cpt_destroyer_tot=cpt_destroyer_tot+cpt_destroyer
            cpt_fregate_tot=cpt_fregate_tot+cpt_fregate
            cpt_lcs_tot=cpt_lcs_tot+cpt_lcs
            cpt_patrol_tot=cpt_patrol_tot+cpt_patrol
            cpt_submarine_tot=cpt_submarine_tot+cpt_submarine
            cpt_supply_tot=cpt_supply_tot+cpt_supply
            cpt_tender_tot=cpt_tender_tot+cpt_tender
            cpt_pha_tot=cpt_pha_tot+cpt_pha
                
            #remise à zéro des compteur
            
            cpt_amphiby=0
            cpt_barge=0
            cpt_carrier=0
            cpt_cruiser=0
            cpt_destroyer=0
            cpt_fregate=0
            cpt_lcs=0
            cpt_patrol=0
            cpt_submarine=0
            cpt_supply=0
            cpt_tender=0
            cpt_pha=0 
    else:   
        for elt in train:
          L_train.append(elt.split('.')[0].split('/')[-1])
          
        for elt in L_train:   
            tree = ET.parse("VOC2007/Annotations/"+str(elt)+'.xml')
            root = tree.getroot()
            
            
            for object_elem in root.iter('object'):
                name = object_elem.find('name').text
                
                if name=='carrier':
                    cpt_carrier+=1
                    
                elif name=='cruiser':
                    cpt_cruiser+=1
                elif name=='tender':
                    cpt_tender+=1
                elif name=='amphiby':
                    cpt_amphiby+=1
                elif name=='patrol':
                    cpt_patrol+=1
                elif name=='destroyer':
                    cpt_destroyer+=1
                elif name=='littoral combat ship':
                    cpt_lcs+=1
                elif name=='submarine':
                    cpt_submarine+=1
                elif name=='supply':
                    cpt_supply+=1
                elif name=='fregate':
                    cpt_fregate+=1
                elif name=='barge':
                    cpt_barge+=1
                elif name=='pha':
                    cpt_pha+=1
        
            
            if cpt_carrier!=0:
                cptI_carrier+=1
            if cpt_cruiser!=0:
                cptI_cruiser+=1
            if cpt_tender!=0:
                cptI_tender+=1
            if cpt_amphiby!=0:
                cptI_amphiby+=1
            if cpt_patrol!=0:
                cptI_patrol+=1
            if cpt_destroyer!=0:
                cptI_destroyer+=1
            if cpt_lcs!=0:
                cptI_lcs+=1
            if cpt_submarine!=0:
                cptI_submarine+=1
            if cpt_supply!=0:
                cptI_supply+=1
            if cpt_fregate!=0:
                cptI_fregate+=1
            if cpt_barge!=0:
                cptI_barge+=1
            if cpt_pha!=0:
                cptI_pha+=1
            
            
            #increment compteur occurence
            cpt_amphiby_tot=cpt_amphiby_tot+cpt_amphiby
            cpt_barge_tot=cpt_barge_tot+cpt_barge
            cpt_carrier_tot=cpt_carrier_tot+cpt_carrier
            cpt_cruiser_tot=cpt_cruiser_tot+cpt_cruiser
            cpt_destroyer_tot=cpt_destroyer_tot+cpt_destroyer
            cpt_fregate_tot=cpt_fregate_tot+cpt_fregate
            cpt_lcs_tot=cpt_lcs_tot+cpt_lcs
            cpt_patrol_tot=cpt_patrol_tot+cpt_patrol
            cpt_submarine_tot=cpt_submarine_tot+cpt_submarine
            cpt_supply_tot=cpt_supply_tot+cpt_supply
            cpt_tender_tot=cpt_tender_tot+cpt_tender
            cpt_pha_tot=cpt_pha_tot+cpt_pha
                
            #remise à zéro des compteur
            
            cpt_amphiby=0
            cpt_barge=0
            cpt_carrier=0
            cpt_cruiser=0
            cpt_destroyer=0
            cpt_fregate=0
            cpt_lcs=0
            cpt_patrol=0
            cpt_submarine=0
            cpt_supply=0
            cpt_tender=0
            cpt_pha=0 

    return([[cptI_amphiby,cpt_amphiby_tot],[cptI_barge, cpt_barge_tot],[cptI_carrier,cpt_carrier_tot],[cptI_cruiser,cpt_cruiser_tot],[ cptI_destroyer,cpt_destroyer_tot],[cptI_fregate,cpt_fregate_tot],[cptI_lcs,cpt_lcs_tot],[cptI_patrol,cpt_patrol_tot],[cptI_submarine,cpt_submarine_tot],[ cptI_supply,cpt_supply_tot],[cptI_tender,cpt_tender_tot],[  cptI_pha,cpt_pha_tot]])    
    
