import shutil, zipfile, os, sys

sys.path.append(os.path.abspath("../"))

from dotenv import load_dotenv

load_dotenv("../.env.dev")


def main():
    print("Ejecutando pruebas básicas de los módulos...")

    try:
        import apps_etiquetado.convertor_cordenadas
        import apps_etiquetado.procesador_anotaciones_cvat
        import apps_etiquetado.procesador_coco_dataset
        import apps_etiquetado.utils_coco_dataset
        import apps_etiquetado.procesador_recortes
        import apps_etiquetado.visualizador_coco_dataset

        print("Todos los módulos se importaron correctamente.")
    except ImportError as e:
        print(f"Error al importar un módulo: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")


if __name__ == "__main__":
    main()
