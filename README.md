# lsvtest

lsvtest es un paquete Python que proporciona funcionalidades para la detección y reconocimiento de LSV (Lenguaje de Señas Venezolano).

## Instalación

Puedes instalar `lsvtest` utilizando pip:

```bash
pip install lsvtest
```

Uso
Para usar lsvtest, sigue estos pasos:

Crear una estructura de proyectoCrea una estructura de proyecto como la siguiente:

```
project/
│
├── main.py
└── script.py
```

Contenido de main.py

```python
from flask import Flask, jsonify
import subprocess

app = Flask(__name__)

# Variable to keep track of the process
process = None

@app.route("/start", methods=["GET"])
def start_process():
    global process
    if process is None:
        # Replace 'your_command_here' with the command you want to run
        process = subprocess.Popen(
            [
                "python",
                "-c",
                'import sys; sys.path.append("."); import script; script.start()',
            ]
        )
        return jsonify({"status": "Process started"}), 200
    else:
        return jsonify({"status": "Process is already running"}), 200

@app.route("/stop", methods=["GET"])
def stop_process():
    global process
    if process is not None:
        process.terminate()  # Sends SIGTERM
        process = None
        return jsonify({"status": "Process stopped"}), 200
    else:
        return jsonify({"status": "No process is running"}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

Contenido de script.py

```python
from lsvtest import LSVRecognition

def start():
    recognition_service = LSVRecognition()

    recognition_service.continuous_detection(
        source=0, 
        output=lambda token: print(token), 
        wsl_compatibility=True, 
        show_video=True
    )
```

Ejecuta el servidor Flask desde la terminal en la carpeta del proyecto:
```bash
python main.py
```

Envía una solicitud GET a http://localhost:5000/start para iniciar el proceso de reconocimiento.

Envía una solicitud GET a http://localhost:5000/stop para detener el proceso de reconocimiento.

Puedes personalizar el reconocimiento modificando los parámetros de continuous_detection en script.py.

## Problemas al instalar

Si tienes problemas al momento de instalar el paquete, prueba ejecutar primero:

```bash
pip install mediapipe opencv-python scikit-learn  tensorflow  
```

Y luego prueba a instalar el paquete nuevamente