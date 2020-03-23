from sklearn.feature_extraction.text import CountVectorizer
from os import listdir, path
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

class Modelo(object):
    def __init__(self, sources = None):
        self.sources = sources
        self.data = DataFrame({'text': [], 'label': []})

    def entrenar(self):
        for ubicacion, label in self.sources:
            # Añadir cada DataFrame que devuelve la función build_data_frame dada la ruta de la carpeta y su etiqueta correspondiente.
            print("Creando dataframe...")
            self.data = self.data.append(self.buildDataFrame(ubicacion, label))
            self.data = self.data.reindex(np.random.permutation(self.data.index)) #permutar los datos
        self.extraerCaracteristicas()
        self.entrenar_modelo()

    def read_files(self, ubicacion): #ubicacion es el nombre de la carpeta que contiene el archivo
        lista_de_nombres = listdir(ubicacion)# Obtener los nombres de todos los ficheros.
        i = 0
        for nombre_de_archivo in lista_de_nombres: # Recorrer los ficheros.
            ruta_de_archivo = path.join(ubicacion, nombre_de_archivo) # Obtener la ruta del fichero, se guarda en file_patch
            with open(ruta_de_archivo, encoding="latin-1") as f:
                lines = f.readlines()
            text = '\n'.join(lines) # Unir las líneas
            i+=1
            print(str(i) + "/" + str(len(lista_de_nombres)) + "/" + ubicacion, end="\r")
            yield ruta_de_archivo, text # Devolver la ruta del fichero y el texto.

    def buildDataFrame(self, ubicacion, label):
        rows = []
        index = []
        for ruta_de_archivo, text in self.read_files(ubicacion): # Recorrer todos los ficheros dada un ruta.
            rows.append({'text': text, 'label': label}) # Añadir el texto del fichero junto con su etiqueta correspondiente.
            index.append(ruta_de_archivo)# Utilizar la ruta del fichero como index.
    
        return DataFrame(rows, index=index) # Crear un nuevo DataFrame a partir de las filas y los indexes.
    
    def extraerCaracteristicas(self):
        print("Extrayendo caracteristicas...")
        self.count_vectorizer = CountVectorizer()
        cuerpos_de_texto = self.data['text'].values # Los textos que tenemos en el DataFrame.
        features = self.count_vectorizer.fit_transform(cuerpos_de_texto) # Aprender el vocabulario del corpus y extraer el recuento de tokens.
        labels = self.data['label'].values #Las etiquetas que hay en el DataFrame
        self.features_train, features_test, \
        self.labels_train, labels_test = train_test_split(
            features, labels,
            train_size=0.8,
            test_size=0.2,
            random_state=0)
        
        return features_test, labels_test
    
    def entrenar_modelo(self):
        print("Entrenando modelo...")
        self.clf = MultinomialNB().fit(self.features_train, self.labels_train)
        joblib.dump(self.clf, "modelos/modeloEntrenado.pkl")
        joblib.dump(self.count_vectorizer, "modelos/vectorizer.pkl")

    def predecir(self, texto):
        #read .pkl file
        try:
            self.clf = joblib.load("modelos/modeloEntrenado.pkl") #cargar un modelo ya entrenado
            self.vectorizer = joblib.load("modelos/vectorizer.pkl")
        except Exception as error:
            print(error)
        # Tokenización.
        features_text = self.vectorizer.transform([texto])
        # Predecir etiquetas.
        prediccion = self.clf.predict(features_text)
        return prediccion