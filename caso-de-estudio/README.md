# Modelo de Regresión Lineal para predecir el precio de la vivienda
# Predicción de Precios de Vivienda

Este proyecto utiliza un modelo de Machine Learning para predecir el precio de viviendas basado en diversos factores. Se ha desarrollado un notebook en Jupyter para entrenar y probar el modelo, así como una API con FastAPI para realizar consultas.

## Requisitos Previos

Asegúrate de tener instalado Python 3.8 o superior. Además, necesitarás `virtualenv` para gestionar entornos virtuales.

## Instalación y Configuración

### 1. Crear un Entorno Virtual
Ejecuta el siguiente comando en la raíz del proyecto:

```bash
python -m venv venv
```

### 2. Activar el Entorno Virtual
- **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

## Ejecución del Notebook

El notebook se encuentra en la carpeta `casos-de-estudio`. Para ejecutarlo, usa el siguiente comando:

```bash
jupyter notebook casos-de-estudio/spanish_houses.ipynb
```

Ejecuta todas las celdas del notebook y guarda los datos generados en la última celda, ya que serán utilizados en la consulta a la API.

## Iniciar la API con FastAPI

Navega a la carpeta de la API:

```bash
cd casos-de-estudio/api
```

Ejecuta el servidor de desarrollo con FastAPI:

```bash
uvicorn main:app --reload
```

## Realizar una Consulta al Endpoint

Utiliza los datos obtenidos en la última celda del notebook para realizar una consulta al endpoint. Puedes hacer una petición con `curl` o Postman. Ejemplo con `curl`:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"feature1": valor1, "feature2": valor2, ...}'
```

También puedes acceder a la documentación interactiva de FastAPI en:

```
http://127.0.0.1:8000/docs
```

## Contacto
Para dudas o sugerencias, puedes abrir un issue en este repositorio.

