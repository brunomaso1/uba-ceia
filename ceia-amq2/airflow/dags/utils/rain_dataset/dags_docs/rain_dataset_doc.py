DESCRIPTION = "ETL: Proceso ETL para el modelo Rain"

FULL_DESCRIPTION_MD = f"""
# ETL -> Rain Model

## Descripción

{DESCRIPTION}

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