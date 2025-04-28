import os
import re
from typing import Any, Dict, Optional

import yaml
from yaml.parser import ParserError


class Config:
    """Clase para cargar y manejar configuraciones desde archivos YAML con soporte para variables de entorno."""

    def __init__(self):
        self._var_matcher = re.compile(r"\${([^}^{]+)}")
        self._tag_matcher = re.compile(r"[^$]*\${([^}^{]+)}.*")
        
        # Configurar el cargador de YAML
        yaml.add_implicit_resolver("!envvar", self._tag_matcher, None, yaml.SafeLoader)
        yaml.add_constructor("!envvar", self._path_constructor, yaml.SafeLoader)
        
        self._config_data: Dict = {}
    
    def _path_constructor(self, _loader: Any, node: Any):
        """Constructor para reemplazar variables de entorno en el YAML."""
        def replace_fn(match):
            envparts = f"{match.group(1)}:".split(":")
            return os.environ.get(envparts[0], envparts[1])
        return self._var_matcher.sub(replace_fn, node.value)
    
    def load_yaml(self, filename: str) -> Dict:
        """Carga un archivo YAML con soporte para variables de entorno.
        
        Args:
            filename: Ruta al archivo YAML.
            
        Returns:
            Diccionario con la configuración cargada, o diccionario vacío en caso de error.
        """
        try:
            with open(filename, "r") as f:
                self._config_data = yaml.safe_load(f.read())
                return self._config_data
        except (FileNotFoundError, PermissionError, ParserError):
            self._config_data = {}
            return self._config_data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor de la configuración.
        
        Args:
            key: Clave a buscar.
            default: Valor por defecto si la clave no existe.
            
        Returns:
            Valor de la configuración o el valor por defecto.
        """
        return self._config_data.get(key, default)
    
    @property
    def config(self) -> Dict:
        """Propiedad para acceder a la configuración completa."""
        return self._config_data
    
    # Implementación de acceso tipo diccionario
    def __getitem__(self, key):
        """Permite acceder a las claves de configuración como un diccionario."""
        return self._config_data[key]
    
    def __contains__(self, key):
        """Permite usar el operador 'in' para verificar si una clave existe."""
        return key in self._config_data
    
    def __len__(self):
        """Permite usar len() para obtener el número de claves de primer nivel."""
        return len(self._config_data)
    
    def keys(self):
        """Devuelve las claves de primer nivel de la configuración."""
        return self._config_data.keys()
    
    def values(self):
        """Devuelve los valores de primer nivel de la configuración."""
        return self._config_data.values()
    
    def items(self):
        """Devuelve los pares clave-valor de primer nivel de la configuración."""
        return self._config_data.items()