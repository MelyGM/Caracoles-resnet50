# %% [markdown]
# # Clasificación de familias de caracoles con Transfer Learning
#
# Clasificación de imágenes de caracoles en **6 familias** usando **ResNet50**
# preentrenado en ImageNet.
#
# **Flujo:**
# 1. Carga y exploración del dataset
# 2. Extracción de características (base congelada): dos modelos, SIN y CON
#    *data augmentation*
# 3. *Fine-tuning* de las últimas capas de ResNet50 en ambos modelos
# 4. Resultados: 4 gráficas individuales (loss y accuracy de cada modelo),
#    2 matrices de confusión y 2 gráficas comparativas (loss y accuracy)

# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# En clusters compartidos conviene que TF no reserve toda la memoria de la GPU
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"GPUs detectadas: {len(gpus)}" if gpus else "Sin GPU: se entrenara en CPU")

# %% [markdown]
# ## 1. Carga del dataset

# %%
import zipfile

# CARACOLES.zip debe estar en el mismo directorio que este script
if not os.path.isdir('mis_imagenes/CARACOLES'):
    with zipfile.ZipFile('CARACOLES.zip') as z:
        z.extractall('mis_imagenes')

# %%
clases = sorted(os.listdir('mis_imagenes/CARACOLES'))
print(f"{len(clases)} clases encontradas: {clases}")

# %% [markdown]
# El zip contiene la estructura `CARACOLES/<familia>/*.png`, así que el
# directorio de datos debe ser `mis_imagenes/CARACOLES` para que cada
# subcarpeta (familia) sea una clase distinta.

# %%
directorio_datos = 'mis_imagenes/CARACOLES'

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# %% [markdown]
# ### Dataset de entrenamiento

# %%
train_dataset = tf.keras.utils.image_dataset_from_directory(
    directorio_datos,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

# %% [markdown]
# ### Dataset de validación

# %%
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    directorio_datos,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"Clases de caracoles encontradas ({num_classes}): {class_names}")

for clase in class_names:
    n = len(os.listdir(os.path.join(directorio_datos, clase)))
    print(f"  {clase}: {n} imágenes")

# %% [markdown]
# **Optimizar la carga de datos**

# %%
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# %% [markdown]
# ## 2. Aumento de datos y modelo base

# %%
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# %% [markdown]
# **Preprocesamiento y modelo base**
#
# Cada modelo recibe su **propia instancia de ResNet50**: si compartieran la
# misma base, el fine-tuning de uno modificaría los pesos del otro y la
# comparación sería inválida.

# %%
preprocess_input = tf.keras.applications.resnet50.preprocess_input

IMG_SHAPE = IMG_SIZE + (3,)


def construir_modelo(usar_augmentation):
    base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    # Congelamos el modelo base: solo se entrenará la cabeza clasificadora
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs) if usar_augmentation else inputs
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model, base_model


def crear_callbacks():
    return [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=3,
                                             restore_best_weights=True)]


def aplicar_fine_tuning(model, base_model, ultimas_capas=40):
    # Descongelamos solo las últimas capas y bajamos el LR para no
    # destruir los pesos preentrenados
    base_model.trainable = True
    for layer in base_model.layers[:-ultimas_capas]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    print(f"Pesos entrenables: {len(model.trainable_weights)}")

# %% [markdown]
# ## 3. Extracción de características
#
# ### Modelo SIN data augmentation

# %%
initial_epochs = 10

print("--- Entrenando Modelo SIN Data Augmentation ---")
model_sin_aug, base_sin = construir_modelo(usar_augmentation=False)
history_sin = model_sin_aug.fit(train_dataset,
                                epochs=initial_epochs,
                                validation_data=validation_dataset,
                                callbacks=crear_callbacks())

# %% [markdown]
# ### Modelo CON data augmentation

# %%
print("\n--- Entrenando Modelo CON Data Augmentation ---")
model_con_aug, base_con = construir_modelo(usar_augmentation=True)
history_con = model_con_aug.fit(train_dataset,
                                epochs=initial_epochs,
                                validation_data=validation_dataset,
                                callbacks=crear_callbacks())

# %% [markdown]
# ## 4. Fine-tuning
#
# Descongelamos las **últimas 40 capas** de ResNet50 en **ambos** modelos y
# seguimos entrenando con una tasa de aprendizaje 10x menor. Así la
# comparación SIN vs CON augmentation se hace entre dos modelos con el
# mismo pipeline completo.

# %%
fine_tune_epochs = 10

ep_sin = len(history_sin.history['loss'])
aplicar_fine_tuning(model_sin_aug, base_sin)
print("--- Fine-tuning del modelo SIN Data Augmentation ---")
history_fine_sin = model_sin_aug.fit(train_dataset,
                                   epochs=ep_sin + fine_tune_epochs,
                                   initial_epoch=ep_sin,
                                   validation_data=validation_dataset,
                                   callbacks=crear_callbacks())

# %%
ep_con = len(history_con.history['loss'])
aplicar_fine_tuning(model_con_aug, base_con)
print("--- Fine-tuning del modelo CON Data Augmentation ---")
history_fine_con = model_con_aug.fit(train_dataset,
                                   epochs=ep_con + fine_tune_epochs,
                                   initial_epoch=ep_con,
                                   validation_data=validation_dataset,
                                   callbacks=crear_callbacks())

# %% [markdown]
# ## 5. Resultados
#
# ### Gráficas individuales (train + validación de cada modelo)

# %%
# Historiales completos: extracción de características + fine-tuning
acc_sin = history_sin.history['accuracy'] + history_fine_sin.history['accuracy']
val_acc_sin = history_sin.history['val_accuracy'] + history_fine_sin.history['val_accuracy']
loss_sin = history_sin.history['loss'] + history_fine_sin.history['loss']
val_loss_sin = history_sin.history['val_loss'] + history_fine_sin.history['val_loss']

acc_con = history_con.history['accuracy'] + history_fine_con.history['accuracy']
val_acc_con = history_con.history['val_accuracy'] + history_fine_con.history['val_accuracy']
loss_con = history_con.history['loss'] + history_fine_con.history['loss']
val_loss_con = history_con.history['val_loss'] + history_fine_con.history['val_loss']


def graficar_entrenamiento(train_vals, val_vals, metrica, titulo, archivo, corte_ft):
    plt.figure(figsize=(8, 5))
    plt.plot(train_vals, label='Entrenamiento')
    plt.plot(val_vals, label='Validación')
    plt.axvline(corte_ft - 1, color='gray', linestyle='--', label='Inicio fine-tuning')
    plt.title(titulo)
    plt.xlabel('Épocas')
    plt.ylabel('Precisión' if metrica == 'accuracy' else 'Pérdida')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    plt.show()


# Modelo SIN data augmentation
graficar_entrenamiento(loss_sin, val_loss_sin, 'loss',
                       'Pérdida — modelo SIN data augmentation',
                       'loss_sin_aug.png', ep_sin)
graficar_entrenamiento(acc_sin, val_acc_sin, 'accuracy',
                       'Precisión — modelo SIN data augmentation',
                       'accuracy_sin_aug.png', ep_sin)

# Modelo CON data augmentation
graficar_entrenamiento(loss_con, val_loss_con, 'loss',
                       'Pérdida — modelo CON data augmentation',
                       'loss_con_aug.png', ep_con)
graficar_entrenamiento(acc_con, val_acc_con, 'accuracy',
                       'Precisión — modelo CON data augmentation',
                       'accuracy_con_aug.png', ep_con)

# %% [markdown]
# ### Matrices de confusión

# %%
def evaluar_modelo(model, nombre, archivo):
    y_true, y_pred = [], []
    for images, labels in validation_dataset:
        logits = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(logits, axis=1))

    print(f"\n=== Reporte de clasificación — modelo {nombre} ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title(f'Matriz de confusión — modelo {nombre}')
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    plt.show()


evaluar_modelo(model_sin_aug, 'SIN data augmentation', 'matriz_sin_aug.png')
evaluar_modelo(model_con_aug, 'CON data augmentation', 'matriz_con_aug.png')

# %% [markdown]
# ### Comparación conjunta (ambos modelos en la misma gráfica)

# %%
epocas_sin = range(1, len(loss_sin) + 1)
epocas_con = range(1, len(loss_con) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epocas_sin, val_loss_sin, 'r--', linewidth=2, label='Validación SIN Aumento')
plt.plot(epocas_con, val_loss_con, 'g-', linewidth=2, label='Validación CON Aumento')
plt.plot(epocas_sin, loss_sin, 'r', alpha=0.3, label='Entrenamiento SIN')
plt.plot(epocas_con, loss_con, 'g', alpha=0.3, label='Entrenamiento CON')
plt.title('Comparación de Pérdida (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('comparacion_loss.png', dpi=150)
plt.show()

# %%
plt.figure(figsize=(8, 5))
plt.plot(epocas_sin, val_acc_sin, 'r--', linewidth=2, label='Validación SIN Aumento')
plt.plot(epocas_con, val_acc_con, 'g-', linewidth=2, label='Validación CON Aumento')
plt.plot(epocas_sin, acc_sin, 'r', alpha=0.3, label='Entrenamiento SIN')
plt.plot(epocas_con, acc_con, 'g', alpha=0.3, label='Entrenamiento CON')
plt.title('Comparación de Precisión (Accuracy)')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('comparacion_accuracy.png', dpi=150)
plt.show()

# %% [markdown]
# ### Ejemplos de predicción (modelo CON data augmentation)

# %%
plt.figure(figsize=(12, 12))
for images, labels in validation_dataset.take(1):
    preds = np.argmax(model_con_aug.predict(images, verbose=0), axis=1)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        pred, real = int(preds[i]), int(labels[i])
        color = 'green' if pred == real else 'red'
        plt.title(f"Pred: {class_names[pred]}\nReal: {class_names[real]}", color=color, fontsize=9)
        plt.axis("off")
plt.tight_layout()
plt.savefig('predicciones_ejemplo.png', dpi=150)
plt.show()

# %%
model_con_aug.save('clasificador_caracoles.keras')
print("Modelo guardado: clasificador_caracoles.keras")
