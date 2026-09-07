from abc import ABC, abstractmethod
from typing import BinaryIO, Optional, Union


class StorageBackend(ABC):
    """Abstract interface for file storage backends (Local disk, S3, GCS, etc.)."""

    @abstractmethod
    def save(self, content: Union[bytes, BinaryIO], filename: str) -> str:
        """
        Persist file content and return the relative stored path or key.
        """
        pass

    @abstractmethod
    def read(self, filename: str) -> bytes:
        """
        Read the entire binary contents of a stored file.
        """
        pass

    @abstractmethod
    def get_path(self, filename: str) -> str:
        """
        Get the absolute or localized path to the stored file.
        """
        pass

    @abstractmethod
    def delete(self, filename: str) -> bool:
        """
        Delete a stored file. Returns True if deleted or already absent, False on failure.
        """
        pass

    @abstractmethod
    def exists(self, filename: str) -> bool:
        """
        Check if a stored file exists.
        """
        pass
