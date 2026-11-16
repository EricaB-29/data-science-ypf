# Equipo 2

### Integrantes

* Erica Blanco
* Francesca Matias

### Tema elegido

*Churn*

### Pregunta a responder

> ¿Podemos predecir si un cliente va a abandonar el banco (Exited) a partir de sus datos demográficos y de cuenta?

### Origen del dataset

*Kaggle*

Dataset: [Churn Modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling/data)

### ¿Qué contiene el dataset?

El dataset contiene información detallada sobre los clientes de un banco y diferentes características demográficas y relacionadas con sus cuentas.

### Campos

| Campo             | Descripción                                                                |
| ----------------- | -------------------------------------------------------------------------- |
| CustomerId      | Identificador único del cliente.                                           |
| Surname         | Apellido del cliente.                                                      |
| CreditScore     | Puntaje crediticio del cliente.                                            |
| Geography       | País al que pertenece el cliente.                                          |
| Gender          | Género del cliente: femenino o masculino.                                  |
| Age             | Edad del cliente.                                                          |
| Tenure          | Cantidad de años que el cliente ha estado en el banco.                     |
| Balance         | Saldo bancario del cliente.                                                |
| NumOfProducts   | Cantidad de productos bancarios que utiliza el cliente.                    |
| HasCrCard       | Indica si el cliente tiene una tarjeta de crédito con el banco.            |
| IsActiveMember  | Indica si el cliente es un miembro activo del banco.                       |
| EstimatedSalary | Salario estimado del cliente.                                              |
| Exited          | Variable binaria: 1 si el cliente abandonó el banco y 0 si permaneció. |

### Resumen del proyecto

Trabajamos con el dataset Churn Modelling, que tiene información de 10.000 clientes de un banco, para estudiar por qué algunos clientes abandonan la entidad. Recorrimos el proceso completo de un proyecto de datos: análisis exploratorio, preparación de las variables, modelos supervisados de clasificación para predecir el abandono y un modelo de clustering (aprendizaje no supervisado) para segmentar a los clientes y ver qué perfiles tienen más riesgo de irse.

### Objetivo

Construir un modelo que prediga si un cliente abandona el banco (Exited) a partir de sus datos demográficos y de cuenta, evaluarlo de forma confiable evitando el overfitting, y lograr que detecte la mayor cantidad posible de clientes que se van, que son los que al banco le interesa retener. Como objetivo complementario, segmentar a los clientes en grupos con perfiles distintos y ver cuáles tienen más abandono.

### Qué descubrimos en el análisis exploratorio

Solo alrededor del 20% de los clientes abandonó el banco, por lo que las clases están desbalanceadas y la exactitud sola no alcanza para evaluar un modelo: uno que siempre responda "se queda" acierta cerca del 80%. Los clientes que se fueron tienen en promedio más edad, y los que tienen saldo en su cuenta se van más que los que tienen saldo cero (Balance tiene dos grupos: uno grande en 0 y otro con saldos altos). Los clientes de Alemania abandonan más que los de otros países, las mujeres más que los hombres (con una diferencia menor) y los clientes inactivos casi el doble que los activos. Además, Francia tiene el doble de clientes que Alemania o España, la gran mayoría usa 1 o 2 productos, y RowNumber, CustomerId y Surname son identificadores que no aportan información predictiva, por lo que se descartaron.

### Modelos ajustados

*Preparación.* Separamos los datos en 80% para entrenamiento y 20% para test, de forma estratificada para mantener la proporción de clientes que se van. La codificación (OneHotEncoder) y el escalado (StandardScaler) se hacen dentro de un Pipeline, después de separar los datos, para evitar fuga de información. Elegimos usar Age y Balance continuas en lugar de sus versiones agrupadas, porque con Random Forest rindieron mejor (AUC en validación cruzada de 0.853 contra 0.826), y agrupamos NumOfProducts en 1, 2 y "3 o más". Todos los modelos se evaluaron con validación cruzada de 5 folds y se compararon contra un piso (DummyClassifier) que siempre predice "se queda".

*Regresión logística.* Fue el primer modelo. No mostró overfitting (AUC de 0.833 en train y 0.831 en validación). Con GridSearchCV el mejor valor fue C=0.1, con un AUC de validación de 0.832: la optimización casi no lo modificó.

*Random Forest.* Con los hiperparámetros por defecto memorizó los datos de entrenamiento (AUC de 1.000 en train y 0.853 en validación). Optimizamos con GridSearchCV (max_depth=None, min_samples_leaf=5, n_estimators=200), lo que subió el AUC de validación a 0.863 y bajó el de train a 0.968. Fue el mejor modelo en validación.

*Random Forest con class_weight='balanced' (modelo final).* El Random Forest optimizado detectaba pocos clientes que se van (recall de 0.455 en test). Con class_weight='balanced' el recall subió a 0.717 y el F1 mejoró, a costa de menor precisión (0.541). Elegimos este modelo porque al banco le interesa detectar a los clientes que abandonan.

| Modelo (evaluado en test) | Exactitud | Precisión | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Piso (Dummy) | 0.796 | 0.000 | 0.000 | 0.000 | 0.500 |
| Random Forest optimizado | 0.862 | 0.777 | 0.455 | 0.574 | 0.858 |
| *Random Forest optimizado + balanced* | 0.818 | 0.541 | 0.717 | 0.617 | 0.858 |

*Clustering (K-means).* Segmentamos a los clientes con K-means sobre cuatro variables estandarizadas (Age, Balance, NumOfProducts, IsActiveMember), sin usar Exited. Elegimos k=6 por el mayor coeficiente de silhouette entre 3 y 6 grupos (0.396, contra 0.137 del modelo inicial con 8 variables y k=3). Los grupos con más abandono (31% a 33%, contra 20% del total) son los clientes inactivos con saldo y los clientes mayores; también probamos DBSCAN, pero generó varios grupos con menos del 2% de los clientes y lo descartamos.
