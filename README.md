# projet_traitement-images
---------------------------


# Technique avec roboflow:
---------------------------

## Téléchargement du dataset roboflow:

Si vous voulez télécharger le dataset il vous suffit de taper la commande suivante sur tout les système *nix

```python
curl -L "https://app.roboflow.com/ds/sNlUG8lUDU?key=GCYYmrpJSo" &gt; roboflow.zip; unzip 
roboflow.zip; rm roboflow.zip
```

## Inférence sur Raspberry pi:

Vous aurrez besoin d'un raspberry pi 4 utilisant  une version 64 bit de ubuntu


> installation de Docker:
```python
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

>Démarrer l'API te utiliser les serveurs pour l'inférence.
```python
sudo docker pull roboflow/inference-server:cpu
sudo docker run --net=host roboflow/inference-server:cpu
```

>Installez roboflow
```python
apt install python3-pip #installation de pip
pip install roboflow #installation de roboflow
```



