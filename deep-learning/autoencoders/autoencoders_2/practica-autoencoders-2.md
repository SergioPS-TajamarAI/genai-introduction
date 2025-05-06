### **Práctica: Generación de Rostros con VAE usando CelebA**  
**Objetivo**: Entrenar un Variational Autoencoder (VAE) convolucional para generar rostros nuevos a partir del dataset CelebA.  

---

### **1. Configuración Inicial**  
**Objetivo**: Preparar el entorno y el dataset.  
**Instrucciones**:  
1. **Descargar CelebA**:  
   - El dataset contiene ~200k imágenes de rostros.  
   - Formato recomendado: Descargar la versión alineada y recortada (64x64 píxeles).  
   - *pista*: Usar `tf.keras.utils.get_file` o la biblioteca `opendatasets` para descargar automáticamente.  
2. **Preprocesamiento**:  
   - Redimensionar imágenes a 64x64 píxeles (si no están ya en ese tamaño).  
   - Normalizar píxeles al rango `[0, 1]` o `[-1, 1]` según la función de activación final.  
3. **Crear DataLoader**:  
   - Usar batches de 64-128 imágenes.  
   - *pista*: `tf.data.Dataset.from_tensor_slices` permite crear un pipeline eficiente.  

---

### **2. Diseño del VAE Convolucional**  
**Objetivo**: Construir un VAE con encoder y decoder convolucionales.  

#### **Encoder**:  
- **Capas**:  
  - 3-4 capas convolucionales (`Conv2D`) con activación `LeakyReLU` o `ReLU`.  
  - Reducir dimensiones espaciales progresivamente (ej: 64x64 → 32x32 → 16x16).  
- **Capa Latente**:  
  - Dos salidas: `z_mean` y `z_log_var` (media y log-varianza de la distribución latente).  
  - Dimensión típica del espacio latente: 128-512.  
- *pista*: Usar `Flatten` antes de las capas densas para `z_mean` y `z_log_var`.  

#### **Reparameterization Trick**:  
- Muestrear `z` usando:  
  ```  
  z = z_mean + exp(z_log_var * 0.5) * epsilon  
  ```  
  donde `epsilon ~ N(0, 1)`.  

#### **Decoder**:  
- **Capas**:  
  - 3-4 capas `Conv2DTranspose` o `UpSampling2D + Conv2D` para aumentar resolución.  
  - Usar activación `sigmoid` en la última capa si las imágenes están en `[0, 1]`.  
- *pista*: Asegurar que la salida final tenga las mismas dimensiones que la entrada (64x64x3).  

---

### **3. Función de Pérdida**  
**Objetivo**: Definir la pérdida del VAE (reconstrucción + divergencia KL).  
1. **Reconstrucción**:  
   - Error cuadrático medio (MSE) entre imágenes originales y reconstruidas.  
   - *Alternativa*: Pérdida de entropía cruzada binaria (BCE) si se usa `sigmoid`.  
2. **Divergencia KL**:  
   - Calculada entre la distribución latente y una normal estándar:  
     ```  
     KL = -0.5 * sum(1 + z_log_var - z_mean^2 - exp(z_log_var))  
     ```  
3. **Pérdida Total**: `loss = reconstruction_loss + beta * KL_loss` (beta=1 por defecto).  
   - *pista*: Usar `beta` como hiperparámetro para ajustar el trade-off.  

---

### **4. Entrenamiento**  
**Objetivo**: Entrenar el modelo y monitorear la generación.  
**Instrucciones**:  
1. **Compilar el modelo**:  
   - Optimizador: `Adam` con learning rate=0.0005.  
   - Métricas opcionales: Seguir `loss`, `reconstruction_loss`, `KL_loss` por separado.  
2. **Callbacks útiles**:  
   - `ModelCheckpoint`: Guardar el mejor modelo.  
   - `TensorBoard`: Visualizar curvas de pérdida.  
3. **Entrenar**:  
   - Épocas: 30-50 (CelebA requiere más tiempo que CIFAR-10).  
   - Batch size: 64-128 (depende de la memoria de GPU).  
4. **Monitoreo visual**:  
   - Generar rostros nuevos cada 5 épocas muestreando `z ~ N(0, 1)`.  

---

### **5. Generación y Evaluación**  
**Objetivo**: Crear rostros nuevos y evaluar la calidad.  
**Instrucciones**:  
1. **Generación**:  
   - Muestrear vectores `z` de `N(0, I)` y pasarlos por el decoder.  
   - *pista*: Si el espacio latente es 2D, se puede visualizar una cuadrícula de rostros variando `z`.  
2. **Evaluación cualitativa**:  
   - ¿Los rostros generados son diversos y realistas?  
   - ¿Existe "mode collapse" (todas las generaciones son similares)?  
3. **Evaluación cuantitativa (opcional)**:  
   - Métricas como **FID** (Fréchet Inception Distance) comparan distribuciones de imágenes reales y generadas.  

---

### **6. Análisis Crítico**  
**Preguntas clave**:  
1. ¿Cómo afecta la dimensión del espacio latente a la generación?  
2. ¿Qué ocurre si aumentamos `beta` (peso de la pérdida KL)?  
3. ¿Por qué las imágenes generadas pueden verse borrosas?  

---

### **7. Entrega**  
**Entregables**:  
- **Código** del VAE (estructura clara y comentada).  
- **Reporte** con:  
  - Gráficos de pérdida durante el entrenamiento.  
  - Muestras de rostros generados en diferentes épocas.  
  - Análisis de los resultados y posibles mejoras.  

---

### **Pistas para Problemas Comunes**  
- **Problema**: Las generaciones son ruidosas o sin sentido.  
  - *Solución*: Aumentar la capacidad del decoder (más filtros/capas).  
- **Problema**: Las imágenes son borrosas.  
  - *Solución*: Usar `Conv2DTranspose` en lugar de `UpSampling2D` o añadir capas residuales.  
- **Problema**: El modelo no converge.  
  - *Solución*: Reducir el learning rate o inicializar las capas convolucionales con `HeNormal`.  

---

### **Recursos Adicionales:**  
- **Dataset CelebA**: [Enlace](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).  
- **Repositorio de referencia**: [VAE en Keras](https://keras.io/examples/generative/vae/). 

### **Ejemplos en internet**
https://colab.research.google.com/github/goodboychan/goodboychan.github.io/blob/main/_notebooks/2021-09-14-03-Variational-AutoEncoder-Celeb-A.ipynb
https://linux-blog.anracom.com/2022/10/22/variational-autoencoder-with-tensorflow-xi-image-creation-by-a-vae-trained-on-celeba/
https://github.com/Ciph3r007/VAE-for-CelebA/blob/main/VAE_for_CelebA_Dataset.ipynb
https://medium.com/the-generator/a-basic-variational-autoencoder-in-pytorch-trained-on-the-celeba-dataset-f29c75316b26
https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

