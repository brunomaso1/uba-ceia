import os, re, yaml
from typing import Any, Dict, Optional
from yaml.parser import ParserError
from utils import Singleton

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
    yaml_filename = None

    def __init__(self, yaml_filename: str):
        self.yaml_filename = yaml_filename
        try:
            with open(yaml_filename, "r") as f:
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
