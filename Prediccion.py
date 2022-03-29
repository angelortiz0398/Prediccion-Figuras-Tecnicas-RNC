import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Importaremos keras para la RNC

longitud, altura = 21, 28

# Cargaremos el modelo pre-entrenado que creamos, junto con sus pesos 
modelo = './Figuras.h5'
pesos_modelo = './pesosF.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)
print("===== Proyecto Predicción del precio de acciones mediante figuras técnicas de trading usando RNC =====")
print("===== Materia IA/SE =====")

# Le diremos el nombre del archivo con el que hara la prediccion. Use los que estan dentro de la carpeta. 
archivo = input("Escribe el nombre del archivo junto con su extension: ")
print("")

def prediccion(file):
  # Importa la imagen y la convierte en un arreglo 
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  # Realiza la prediccion y arroja el resultado de esta forma [1,0,0] con la categoria a donde pertenece nuestra imagen
  array = cnn.predict(x)
  result = array[0]
  respuesta = np.argmax(result)
  if respuesta == 0:
    print(" La prediccion es que es: Cambios en V")
    print(" Deberia comprar porque probablemente su precio incremente")
  elif respuesta == 1:
    print(" La prediccion es que es: Doble o Triple Suelo")
    print(" Deberia comprar porque esta figura se produce al final de una tendencia bajista y suponen el final de la misma")
  elif respuesta == 2:
    print(" La prediccion es que es: Doble o Triple Techo")
    print(" Deberia vender porque esta figura se forma tras una tendencia alcista y marcan su final.")
  elif respuesta == 3:
    print(" La prediccion es que es: Hombro Cabeza Hombro")
    print(" Deberia vender porque esta figura pronostica que cambiara la tendencia a bajista despues del tercer pico")
  elif respuesta == 4:
    print(" La prediccion es que es: Suelo o Techo Redondeado")
    print(" Si es techo redondeado deberia vender porque probablemente cambia la tendencia a bajista, si es piso redondeado es lo opuesto")
  elif respuesta == 5:
    print(" La prediccion es que es: Hombro Triángulos Ascendentes y Descendentes")
    print(" El patrón se forma cuando el nivel de resistencia permanece plano y el nivel de soporte aumenta. Debes mantener para aumentar tus ganancias")
  elif respuesta == 6:
    print(" La prediccion es que es: Taza con Asa")
    print(" Deberia manter porque esta figura pronostica que seguira la tendencia alcista")
  elif respuesta == 7:
    print(" La prediccion es que es: Superiores e Inferiores de Diamante")
    print(" El patron sugiere una reversión bajista que puede desencadenar un movimiento de mercado de tendencia bajista")
  elif respuesta == 8:
    print(" La prediccion es que es: Bandera")
    print(" Es un patrón que se asemeja a un paralelogramo y afirma la tendencia que se lleva actualmente. En caso de ser bajista podria vender, sino es mejor mantener")
  elif respuesta == 9:
    print(" La prediccion es que es: Rectángulo Inferior")
    print(" Comienza en un movimiento de tendencia bajista. Se forma cuando el precio rebota entre el soporte paralelo y las líneas de tendencia de resistencia.")
  return respuesta
prediccion(archivo)
input("ok")