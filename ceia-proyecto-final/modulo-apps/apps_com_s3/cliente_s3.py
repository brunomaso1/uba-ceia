import sys, os
from typing import Any, Dict, Optional

sys.path.append(os.path.abspath("../"))

from apps_utils.utils import Singleton

import boto3
from botocore.exceptions import ClientError

from apps_config.settings import Config

CONFIG = Config().config_data


class S3Client(metaclass=Singleton):
    client = None

    def __init__(self, config: Optional[Dict[str, Any]] = CONFIG["minio"]) -> None:
        try:
            self.client = boto3.client(
                "s3",
                endpoint_url=config["endpoint_url"],
                aws_access_key_id=config["access_key"],
                aws_secret_access_key=config["secret_key"],
                region_name=config["region"],
            )
        except ClientError as e:
            raise Exception(f"Error al conectar a S3: {e}")
