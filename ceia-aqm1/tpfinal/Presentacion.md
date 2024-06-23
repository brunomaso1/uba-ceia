# División del trabajo?
- Metodología
- CRISP-ML(Q)
- Mejora continua

# Que es CRISP-ML(Q)?
Es un marco de trabajo reciente (2021) que aplica las mejores práctias de la ingeniería de software a proyectos orientados a Machine Learning. Es una mejora con respecto a CRISP-DM (estándar ámpliamnete usado en la industria), al cual se le agrega un fuerte enfoque en la gestión de riesgos y el aseguramiento de la calidad.
- Fases
	1. Analisis del negocio y entendimiento de los datos
	2. Peparación de los datos
	3. Modelado
	4. Evaluación
	5. Despliegue
	6. Monitore y mantenimiento
- QA y analisis de riesgo 
- Iterativo

# Objetivo del proyecto
El objetivo del proyecto es predecir si lloverá o no al día siguiente de un conjunto de datos tomados de diversas estaciones meterológicas de Australia entre los años 2007 y 2017.
De los tipos de análisis vistos en el curso, este entra dentro del análisis predictivo.

# Características del conjunto
El conjunto fue obtenido de kaggle, cuenta con cerca de 150000 observaciones, con 23 features. En sí 22 más la variable objetivo.
Las características principales de las features es que hay 4 columnas cualitativas nominales, una columna de fecha, una columna de ubicación geográfica y el resto de columnas cuantitativas continuas. También destacar que la variable objetivo es booleana por lo que este problema entra dentro de la clasificación de binaria, o sea, aprendizaje supervisado.

# Problemas del conjunto
Como problemas principales, podemos ver que hay muchas columnas que les faltan datos y muchas que presentan características de distribuciones no normales. Tambien hay casos donde hay columnas con muchos outliers.

# Que soluciones brindamos
Segumimos los pasos del marco de trabajo de CRISP-ML(Q), en donde en la parte de ententimiento de los datos realizamos las siguientes tareas:
- Primeramente analizamos los atributos desde un punto de vista del negocio, comprendiendo las unidades, y los valores que podría tomar. Por ejemplo: No es sensato que exista un valor máximo de temperatura cercano a los 60 grádos, cuando la máxima histórica en Australia fue de 50 grados. No se encontraron anomalias de este tipo en el conjunto.
- Luego analizamos los datos duplicados, pero basandonos en las columas de fecha y ubicación, ya son las que no deberían tener duplicados. No habían observaciones duplicadas.
- Más adelante, tratamos los tipos de datos, convirtiendolos a sus tipos nativos en numpy, prestando atención a cada columna. Destacar que como teníamos una fecha, la convertimos a datetime[ns] sin problemas. También chequeamos que los varlos faltantes estén realmente con el tipo adecuado, o sea, np.nan.
- Luego empezamos a chequear las características de los datos (los momentos y sus gráficos). De esta etapa obtuvimos que si bien algunas columnas presentaban un comportamiento bastante normal, muchas no eran normales, por lo que si en algún momento tendríamos que utilizar medidas centrales, para algunas podríamos utilizar la media y para otras deberíamos utilizar la mediana. También comprobamos que había un gran desbalance (cerca de 3 veces más) para la calse de salida. Esto significa que tenemos que ver técnicas para cuando las clases están desbalanceadas como over-sampling o under-sampling.
- En la etapa siguiente, analizamos los valores faltantes, donde logramos ver que habían columnas que le faltaban más del 40% de los datos.
- En la ante-última etapa, analizamos los valores atípicos y encontramos que hay columnas que tienen muchos a-tipicos. También vimos la diferencia de clasificar las columnas según el rango intercuartílico o la desviación estandar.
- En la útlima etapa, vimos que habían muchos datos que estaban correlacionados; muchos de los cuales entran dentro de la lógica (como la temperatura de mañana no cambia mucho con respecto a la temperatura de la tarde). Un dato importante, es que pair plot podemos observar la correlación lineal entre ciertas variables analizadas anteriormente. Además, se encontró que las variables Humidity3pm y Sunshine resultan sumamente promimsorias para la separación de clases en la variable objetivo. Sin embargo,  tenemos el problema que la variable Sunshine era una de aquellas variables que le faltaban más del 40% de los datos. Conlcuímos que realizar un buen trabajo de imputación de datos faltantes resulta entonces indispensable para obtener una buena performance en nuestro tarea predictiva. Finalmente se analizó una interesante idea sobre la co-relación entre la ubicación, la fecha y los datos faltantes. Dado que ubicaciones cercanas tienen datos similares.


