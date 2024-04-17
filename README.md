# Instrucciones de instalación

Versión recomendada de Python: 3.10

Se recomienda la creación de un virtual environment antes de instalar las dependencias:

Si se tiene conda instalado:

`conda create --name venv`  
`conda activate venv`

O si se desea usar el módulo virtualenv

`python -m virtualenv venv`
`source venv/bin/activate`


- Abrir el cuaderno 'Motor de reconocimiento.ipynb' y ejecutar la primera línea del mismo para instalar todas las dependencias requeridas


# Instrucciones de ejecución

- En el cuarderno 'Motor de reconocimiento.ipynb' Ejecutar todas las celdas una por una hasta la casilla con el texto "Carga del dataset en formato numpy a memoria" (Sin incluir esta última).

- Ejecutar las 2 casillas siguientes al texto "Definición del primer modelo evaluado: LSTM", que son la definición del modelo y el comando que compila al mismo.

- Ejecutar la casilla con el código: `modelLSTM.load_weights('models/LSTM model.h5')`

- Finalmente, ejecutar la penúltima casilla, que está justo luego del texto "Ejecución del Motor de Reconocimiento de la LSV". 