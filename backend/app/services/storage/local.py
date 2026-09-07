import os
import shutil
from typing import BinaryIO, Union
from app.core.config import settings
from app.services.storage.base import StorageBackend


class LocalStorageBackend(StorageBackend):
    """
    Local filesystem storage backend for development and on-prem deployments.
    Stores files in the directory configured by settings.STORAGE_DIR.
    """

    def __init__(self, base_dir: str = settings.STORAGE_DIR):
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def get_path(self, filename: str) -> str:
        """Sanitize filename and return absolute path inside the base directory."""
        safe_name = os.path.basename(filename)
        return os.path.join(self.base_dir, safe_name)

    def save(self, content: Union[bytes, BinaryIO], filename: str) -> str:
        safe_path = self.get_path(filename)
        if isinstance(content, bytes):
            with open(safe_path, "wb") as f:
                f.write(content)
        else:
            with open(safe_path, "wb") as f:
                shutil.copyfileobj(content, f)
        return safe_path

    def read(self, filename: str) -> bytes:
        safe_path = self.get_path(filename)
        if not os.path.exists(safe_path):
            raise FileNotFoundError(f"File {filename} not found in local storage.")
        with open(safe_path, "rb") as f:
            return f.read()

    def delete(self, filename: str) -> bool:
        safe_path = self.get_path(filename)
        if os.path.exists(safe_path):
            try:
                os.remove(safe_path)
                return True
            except OSError:
                return False
        return True

    def exists(self, filename: str) -> bool:
        return os.path.exists(self.get_path(filename))


_local_storage_instance: LocalStorageBackend | None = None


def get_storage_backend() -> StorageBackend:
    """Retrieve or initialize the active storage backend."""
    global _local_storage_instance
    if _local_storage_instance is None:
        _local_storage_instance = LocalStorageBackend()
    return _local_storage_instance
