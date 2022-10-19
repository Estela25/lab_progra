
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#obtencion de los datos de entrenamiento
far_kel = pd.read_csv("FAR_KEL.csv", sep=";")
#print(far_kel)

#datos de entrada y salida
f = far_kel ['Fahrenheit']
k = far_kel['kelvin']

#modelo de entrenamiento
modelo_f_k= tf.keras.Sequential()
modelo_f_k.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

#compliar modelo
modelo_f_k.compile(optimizer=tf.keras.optimizers.Adam(1),loss='mean_squared_error')

hisotiral_f_k = modelo_f_k.fit(f,k, epochs=150, verbose=0)

#conversor
k= modelo_f_k.predict([86])
print("Conversion de Farehrenheit a kelvin: ",k)


