# Documentaicón

DESCRIPTION_ETL = "ETL: Proceso ETL para el modelo Rain"

FULL_DESCRIPTION_MD_ETL = f"""
# ETL -> Rain Model

## Descripción

{DESCRIPTION_ETL}

## Resumen del proceso

### Entrada

- Ninguna, obtiene el conjunto de internet.

## Tareas

- download_raw_data_from_internet
- upload_raw_data_to_S3
- process_target_drop_na
- search_upload_locations
- process_column_types
- create_inputs_pipe
- create_target_pipe
- split_dataset
- fit_transform_pipes

Cantidad: 9

## Salida

Como salida tiene los conjuntos X_tran, X_test, Y_train, Y_test
ya escalados, listos para ser procesados.
"""

DESCRIPTION_OPTIMIZE = "OPTIMIZE: Proceso de creación y optimización para el modelo Rain"

FULL_DESCRIPTION_MD_OPTIMIZE = f"""
# OPTIMIZE -> Rain Model

## Descripción

{DESCRIPTION_OPTIMIZE}

## Resumen del proceso

### Entrada

- Los datos de X_train... desde S3

## Tareas

- load_train_test_dataset
- experiment_creation
- find_best_model
- test_model
- register_model

Cantidad: 5

## Salida

Como salida tiene el modelo creado y optimizado. También está tagueado.
"""

DESCRIPTION_RETRAIN = "RETRAIN: De re-entrenamiento del dataset Rain"

FULL_DESCRIPTION_MD_RETRAIN = f"""
# RETRAIN -> Rain Model

## Descripción

{DESCRIPTION_RETRAIN}

## Resumen del proceso

### Entrada

- El modelo ya entrenado en MLFlow

## Tareas

- train_the_challenger_model
- evaluate_champion_challenge

Cantidad: 2

## Salida

Como salida tiene el modelo reentrenado. Y si es mejor, también tagueado.
"""