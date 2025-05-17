from pathlib import Path
import os, re, yaml, sys
from typing import Any, Dict, Optional
from yaml.parser import ParserError

sys.path.append(os.path.abspath("../"))  # Agregar el directorio padre al path
from mini_apps_utils.utils import Singleton

tag_matcher = re.compile(r".*\${([^}^{]+)}.*")
var_matcher = re.compile(r"\${([^}^{]+)}")


def path_constructor(_loader: Any, node: Any):
    def replace_fn(match):
        envparts = f"{match.group(1)}:".split(":")
        return os.environ.get(envparts[0], envparts[1])

    return var_matcher.sub(replace_fn, node.value)


# Configurar el cargador de YAML
yaml.add_implicit_resolver("!envvar", tag_matcher, None, yaml.SafeLoader)
yaml.add_constructor("!envvar", path_constructor, yaml.SafeLoader)


class Config(metaclass=Singleton):
    config_data: Dict = {}
    yaml_filename = "config.yaml"  # Ruta por defecto

    def __init__(self, yaml_filepath: Optional[Path] = None):
        if yaml_filepath is None:
            # Obtener la ruta al directorio donde se encuentra este archivo (settings.py)
            settigs_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            self.yaml_filepath = settigs_dir / self.yaml_filename  # Ruta por defecto
        else:
            self.yaml_filepath = yaml_filepath
        try:
            with open(self.yaml_filepath, "r") as f:
                config_data = yaml.safe_load(f)
                self.config_data = self.post_process_config(config_data)
        except (FileNotFoundError, PermissionError, ParserError) as e:
            raise Exception(f"Error al cargar el archivo de configuración: {e}")

    def post_process_config(self, data):
        if isinstance(data, dict):
            return {k: self.post_process_config(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.post_process_config(item) for item in data]
        elif isinstance(data, str) and "${" in data:

            def replace_fn(match):
                envparts = f"{match.group(1)}:".split(":")
                return os.environ.get(envparts[0], envparts[1])

            return var_matcher.sub(replace_fn, data)
        else:
            return data

    # Métodos de acceso como diccionario (ajustados para usar métodos estáticos)
    def __getitem__(self, key):
        return self.config_data[key]

    def __contains__(self, key):
        return key in self.config_data

    def __len__(self):
        return len(self.config_data)

    def keys(self):
        return self.config_data.keys()

    def values(self):
        return self.config_data.values()

    def items(self):
        return self.config_data.items()
