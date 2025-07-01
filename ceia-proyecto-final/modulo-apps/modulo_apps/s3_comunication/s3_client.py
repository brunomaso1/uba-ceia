from dataclasses import dataclass, field
import sys, os
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError
from botocore.client import BaseClient

from modulo_apps.config import MinioConfig, config as CONFIG


@dataclass
class S3Client:
    """Clase para manejar la conexión a S3 usando dataclass."""

    s3client_config: Optional[MinioConfig] = field(default_factory=lambda: CONFIG.minio)
    client: BaseClient = field(init=False)

    def __post_init__(self) -> None:
        try:
            self.client = boto3.client(
                "s3",
                endpoint_url=self.s3client_config.endpoint_url,
                aws_access_key_id=self.s3client_config.access_key,
                aws_secret_access_key=self.s3client_config.secret_key,
                region_name=self.s3client_config.region,
            )
        except ClientError as e:
            raise Exception(f"Error al conectar a S3: {e}")


s3client_instance = S3Client()
s3client = s3client_instance.client
