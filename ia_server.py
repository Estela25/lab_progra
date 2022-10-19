#IMPORTAR LAS LIBRERIAS 
from urllib import parse
from http.server import BaseHTTPRequestHandler, HTTPServer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow_datasets as tfds
import pandas as pd

#Codigo de entrenamiento de IA
#obtencion de los datos de entrenamiento
far_kel = pd.read_csv("FAR_KEL.csv", sep=";")


#datos de entrada y salida
f = far_kel ['Fahrenheit']
k = far_kel['kelvin']

#modelo de entrenamiento
modelo_f_k= tf.keras.Sequential()
modelo_f_k.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

#compliar modelo
modelo_f_k.compile(optimizer=tf.keras.optimizers.Adam(1),loss='mean_squared_error')

hisotiral_f_k = modelo_f_k.fit(f,k, epochs=1000, verbose=0)

#conversor
k= modelo_f_k.predict([10])
print("Conversion de Farehrenheit a kelvin: ",k)

#servidor en Python
class servidorBasico(BaseHTTPRequestHandler):
    def do_GET(self):
        print("Peticion recibida por GET")
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write("Hola Mundo desde, GRUPO 4".encode())

    def do_POST(self):
        print("Peticion recibida por POST")
        #obtenemos los datos enviados por AJAX => Asincrono JavaScript y XML
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)
        data = data.decode('utf-8')
        print(data)

        prediccion = modelo_f_k.predict([data])
        print("Predicción:", prediccion)
        
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(str(prediccion[0][0]).encode())
 
       
print("Iniciando el servidor de Python")
servidor = HTTPServer(("localhost", 3004), servidorBasico)
servidor.serve_forever()