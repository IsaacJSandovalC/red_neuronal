import os
import json
import random

import pickle
import numpy as np

import nltk
from nltk.stem import SnowballStemmer

from data import biblioteca

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

model = Sequential()
stemmer = SnowballStemmer('spanish')                  # definicion de stemmer para español
bolsadepalabras = []    # creación de lista vacia para guardar palabras
clases = []             # creacion de lista para guardar etiquetas de la conversación
documents = []          # creación de lista para guardar entrada y su correspondiente etiqueta.

ignore_words = ["?", "¿", "!", "¡", "."]              # Lista de simbolos que se desean eliminar.
training = []           # Creacion de lista vacia de para agregar los vectores construidos en las siguientes lineas.

def guardar_json(datos, filename):
    '''creacion de funcion para guardar 
    diccionario de conocimiento en formato json'''
    archivo = open(filename, "w")
    json.dump(datos, archivo, indent=4)


# Guardado de diccionario de conocimiento en formato json.
guardar_json(biblioteca, 'intents.json')

for intent in biblioteca['intents']:
    clases.append(intent['tag'])
    for pattern in intent['patterns']:
        result = nltk.word_tokenize(pattern)
        bolsadepalabras.extend(result)
        documents.append((result, intent['tag']))

bolsadepalabras = [stemmer.stem(w.lower())for w in bolsadepalabras if w not in ignore_words]
def cleanString(words, ignore_words):
    '''uncion utilizada para limpiar lista de palabras,
     el uso de funciones, evita repetir la innecesaria de codigo'''
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    return words

bolsadepalabras = cleanString(bolsadepalabras, ignore_words)
for doc in documents:
    # obtencion del primer elemento guardado en cada posicion de la lista documents.
    interaccion = doc[0]
    # limpieza del strin "interaccion"
    interaccion = cleanString(interaccion, ignore_words)

    entradacodificada = []  # creacion de la lista vacia llamada "entradacodificada"
    # codificacion de la entrada
    for palabra in bolsadepalabras:
        if palabra in interaccion:
            entradacodificada.append(1)
        else:
            entradacodificada.append(0)
    # codificacion de la etiqueta
    salidacodificada = [0]*len(clases)
    indice = clases.index(doc[1])
    salidacodificada[indice] = 1
    training.append([entradacodificada, salidacodificada])
# conversion de la lista training a un array de numpy
training = np.array(training, dtype=list)
x_train = list(training[:, 0])
y_train = list(training[:, 1])

if os.path.exists("chatbot_model.h5"):
    # preguntar al usuario si desea sobreescribir el archivo existente
    # o elegir no volver a crear el modelo
    user_input = input("El archivo del modelo ya existe, ¿desea sobreescribirlo? (S/N)")
    if user_input.upper() == "S":
        # proceder a crear el modelo
        model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))  # capa oculta -> aprendizaje
        model.add(Dropout(0.5))
        model.add(Dense(len(y_train[0]), activation='softmax'))

        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)  # ,decay=1e-6
        model.compile(loss="categorical_crossentropy",optimizer=sgd, metrics=["accuracy"])

        hist = model.fit(np.array(x_train), np.array(y_train),epochs=300, batch_size=5, verbose=True)
        model.save("chatbot_model.h5", hist) # guarda bolsa de palabras como archivo .pkl
        pickle.dump(bolsadepalabras, open("bolsadepalabras.pkl", "wb")) # guarda lista de clases como archivo .pkl
        pickle.dump(clases, open("classes.pkl", "wb"))
        print("modelo creado")
    else:
        print("El modelo no se ha creado.")
else:
    # proceder a crear el modelo
        model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))  # capa oculta -> aprendizaje
        model.add(Dropout(0.5))
        model.add(Dense(len(y_train[0]), activation='softmax'))

        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)  # ,decay=1e-6
        model.compile(loss="categorical_crossentropy",optimizer=sgd, metrics=["accuracy"])

        hist = model.fit(np.array(x_train), np.array(y_train),epochs=300, batch_size=5, verbose=True)
        model.save("chatbot_model.h5", hist) # guarda bolsa de palabras como archivo .pkl
        pickle.dump(bolsadepalabras, open("bolsadepalabras.pkl", "wb")) # guarda lista de clases como archivo .pkl
        pickle.dump(clases, open("classes.pkl", "wb"))
        print("modelo creado")

model = load_model("chatbot_model.h5")
biblioteca = json.loads(open("intents.json").read())
words = pickle.load(open("bolsadepalabras.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# definir funcion para aplicar tokenization, stemming sobre el string suministrado por el usuario.
def cleanEntrada(entradaUsuario):
    entradaUsuario = nltk.word_tokenize(entradaUsuario)
    entradaUsuario = [stemmer.stem(w.lower())for w in entradaUsuario if w not in ignore_words]
    return entradaUsuario

def convVector(entradaUsuario, bolsadepalabras):
    entradaUsuario = cleanEntrada(entradaUsuario)
    # colocar vector de entrada como ceros
    vectorentrada = [0]*len(bolsadepalabras)
    for palabra in entradaUsuario:              # loop sobre la entrada del usuario
        # verificación si la palabra esta dentro de la bolsa de palabras.
        if palabra in bolsadepalabras:
            # obtanción del indice de la palabra actual, en la bolsa de palabras
            indice = bolsadepalabras.index(palabra)
            # asignación de 1 en el vector de entrada para el indice correspondiente.
            vectorentrada[indice] = 1

    vectorentrada = np.array(vectorentrada)  # conversión a un arreglo numpy
    return vectorentrada

def getResponse(listEtiquetas, biblioteca):
    etiqueta = listEtiquetas[0]['intent']
    listadediccionarios = biblioteca['intents']
    for dicionario in listadediccionarios:
        if etiqueta == dicionario['tag']:
            listaDeRespuestas = dicionario['responses']
            respuesta = random.choice(listaDeRespuestas)
            break
    return respuesta

def gettag(vectorentrada, LIMITE=0):
    vectorsalida = model.predict(np.array([vectorentrada]))[0]
    # cargar los indices y los valores retornados por el modelo
    vectorsalida = [[i, r] for i, r in enumerate(vectorsalida) if r > LIMITE]
    # ordenar salida en funcion de la probabilidad, valor que está contenido en el segundo termino de cada uno de sus elementos.
    vectorsalida.sort(key=lambda x: x[1], reverse=True)
    listEtiquetas = []
    for r in vectorsalida:
        listEtiquetas.append({"intent": clases[r[0]], "probability": str(r[1])})
    return listEtiquetas