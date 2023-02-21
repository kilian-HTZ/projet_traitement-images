from roboflow import Roboflow
import os

# Chemin vers le dossier contenant les images
image_dir = 'images_test/'

rf = Roboflow(api_key="tAEOzLMyNRFWLVyrM3mw")
project = rf.workspace("uca-zd6kn").project("chips-znaxf")
model = project.version(3, local="http://localhost:9001/").model


# Parcourir le dossier
for filename in os.listdir(image_dir):
    # Vérifier que le fichier est une image
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        prediction = model.predict("1.jpg", confidence=40, overlap=30)
        ## get predictions on hosted images
        #prediction = model.predict("YOUR_IMAGE.jpg", hosted=True)
        print(prediction.json())





