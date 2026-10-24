# Clasificación de familias de caracoles con Transfer Learning

Clasificación de imágenes SEM de caracoles en 6 familias
(**Cerithiidae**, **Embrionic_shell**, **Eulimidae**, **Litiopidae**,
**Neritidae**, **Risooinidae**) mediante transfer learning.

El proyecto compara dos dimensiones:

- **Arquitecturas**: VGG16 y ResNet50.
- **Data augmentation**: para cada arquitectura se entrena un modelo
  **SIN** aumento y otro **CON** aumento (rotación, zoom, contraste),
  dando **4 modelos** en total.

## Estructura del repositorio

```
├── README.md
├── VGG16/
│   ├── vgg_caracoles.py          # script para ejecución en clúster (SLURM)
│   ├── VGG_Caracoles_colab.ipynb # notebook original (Colab)
│   ├── caracoles.slrm            # script de entrega SLURM
│   ├── CARACOLES.zip             # dataset
│   └── resultados_vgg/           # gráficas generadas
└── Resnet50/
    ├── Caracoles.py              # script para ejecución en clúster (SLURM)
    ├── Caracoles_colab.ipynb     # notebook original (Colab)
    ├── run_caracoles.slrm        # script de entrega SLURM
    ├── CARACOLES.zip             # dataset
    └── resultados/               # gráficas generadas
```

## Dataset

- Imágenes SEM organizadas en `CARACOLES/<familia>/*.png`
- Dataset **desbalanceado**: la clase minoritaria (`Risooinidae`) tiene
  ~4.5× menos imágenes que la mayoritaria (`Neritidae`). Se compensa con
  `class_weight` calculado solo sobre el conjunto de entrenamiento.
- **Split estratificado 60/20/20** (train/validación/test, `seed=123`):
  `train_test_split` con `stratify` garantiza la misma proporción de cada
  familia en los tres subconjuntos. 

## Metodología

Común a ambas arquitecturas:

1. **Preprocesamiento**: resize a 224×224 + `preprocess_input` propio de
   cada arquitectura.
2. **Cabeza clasificadora**: `GlobalAveragePooling2D → Dropout(0.2) → Dense(6)`.
3. **Extracción de características**: base congelada, 10 épocas,
   `Adam(lr=1e-3 a 1e-4, clipnorm=1.0)`, `class_weight`,
   `EarlyStopping(restore_best_weights=True)` + `ReduceLROnPlateau`.
4. **Fine-tuning**: se descongela la parte final de la base con un LR menor:
   - **VGG16**: bloque `block5` completo, `lr = 2e-5`
   - **ResNet50**: últimas 40 capas, `lr = 1e-5` (las capas BatchNorm se
     mantienen en modo inferencia)
5. **Evaluación**: reporte de clasificación (`precision`, `recall`, `f1`)
   y matriz de confusión sobre el conjunto de **test**.

## Resultados

Los scripts generan por ejecución:

- `distribucion_clases.png` — histograma del dataset
- `loss_{sin,con}_aug.png` / `accuracy_{sin,con}_aug.png` — curvas por modelo
  (con línea punteada marcando el inicio del fine-tuning)
- `comparacion_loss.png` / `comparacion_accuracy.png` — SIN vs CON en una gráfica
- `matriz_{sin,con}_aug.png` — matrices de confusión sobre test
- `predicciones_ejemplo.png` — ejemplos de predicción
- `clasificador_caracoles.keras` — modelo final (versión CON augmentation)
- `classification_report` impreso en el log de SLURM

Accuracy en validación (última corrida):

| Arquitectura | Métrica | SIN augmentation | CON augmentation |
| --- | --- | --- | --- |
| ResNet50 | Accuracy | 99.5 % | 99.2 % |
| ResNet50 | Loss | 0.017 | 0.023 |
| VGG16 | Accuracy | 99.2 % | 99.0 % |
| VGG16 | Loss | 0.019 | 0.018 |
