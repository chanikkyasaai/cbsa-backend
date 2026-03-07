"""
Azure Blob Storage helper for model checkpoint files (.pth).

Configuration is read from the centralised ``app.config.settings`` object
which in turn reads from environment variables:
  AZURE_STORAGE_CONNECTION_STRING  – full connection string (required)
  AZURE_STORAGE_CONTAINER          – blob container name (default: "cbsa-models")

If the connection string is absent, upload/download are silently skipped and
the app falls back to local-disk model files.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import so the app starts even without the Azure SDK
# ---------------------------------------------------------------------------
try:
    from azure.storage.blob import BlobServiceClient  # type: ignore[import]

    _BLOB_SDK_AVAILABLE = True
except ImportError:
    _BLOB_SDK_AVAILABLE = False
    logger.warning(
        "azure-storage-blob package not installed – "
        "Blob Storage model store disabled.  "
        "Install it with: pip install azure-storage-blob"
    )

_LOCAL_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "data" / "checkpoints"


class BlobModelStore:
    """Upload / download .pth checkpoint files to/from Azure Blob Storage."""

    def __init__(self) -> None:
        self._container_client = None
        self._enabled = False
        self._try_connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _try_connect(self) -> None:
        if not _BLOB_SDK_AVAILABLE:
            return

        conn_str = settings.AZURE_STORAGE_CONNECTION_STRING.strip()
        if not conn_str:
            logger.info(
                "AZURE_STORAGE_CONNECTION_STRING not set – "
                "Blob model store disabled"
            )
            return

        container_name = settings.AZURE_STORAGE_CONTAINER

        try:
            blob_service = BlobServiceClient.from_connection_string(conn_str)
            self._container_client = blob_service.get_container_client(
                container_name
            )
            # Create the container if it doesn't exist
            try:
                self._container_client.create_container()
            except Exception:
                pass  # already exists
            self._enabled = True
            logger.info(
                "Azure Blob model store connected: container=%s",
                container_name,
            )
        except Exception as exc:
            logger.error("Failed to connect to Azure Blob Storage: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_model(self, local_path: str, blob_name: str) -> bool:
        """
        Upload a local .pth file to Blob Storage.

        Args:
            local_path: Path to the local .pth file.
            blob_name:  Name of the blob (e.g. ``gat_checkpoint.pt``).

        Returns ``True`` on success, ``False`` if Blob Storage is unavailable.
        """
        if not self._enabled or self._container_client is None:
            logger.debug("Blob upload skipped (store not enabled)")
            return False

        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            with open(local_path, "rb") as fh:
                blob_client.upload_blob(fh, overwrite=True)
            logger.info("Model uploaded to blob: %s", blob_name)
            return True
        except Exception as exc:
            logger.error("Failed to upload model to blob %s: %s", blob_name, exc)
            return False

    def download_model(self, blob_name: str, local_path: str) -> bool:
        """
        Download a .pth file from Blob Storage to a local path.

        Returns ``True`` on success.
        """
        if not self._enabled or self._container_client is None:
            return False

        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            with open(local_path, "wb") as fh:
                stream = blob_client.download_blob()
                fh.write(stream.readall())
            logger.info(
                "Model downloaded from blob %s → %s", blob_name, local_path
            )
            return True
        except Exception as exc:
            logger.error(
                "Failed to download model from blob %s: %s", blob_name, exc
            )
            return False

    def download_model_bytes(self, blob_name: str) -> Optional[bytes]:
        """Download a .pth blob into memory (returns raw bytes or None)."""
        if not self._enabled or self._container_client is None:
            return None
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            stream = blob_client.download_blob()
            return stream.readall()
        except Exception as exc:
            logger.error(
                "Failed to download model bytes from blob %s: %s",
                blob_name,
                exc,
            )
            return None

    def delete_model(self, blob_name: str) -> bool:
        """Delete a single blob."""
        if not self._enabled or self._container_client is None:
            return False
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            logger.info("Blob deleted: %s", blob_name)
            return True
        except Exception as exc:
            logger.error("Failed to delete blob %s: %s", blob_name, exc)
            return False

    def delete_all_models(self) -> int:
        """Delete every blob in the container.  Returns count deleted."""
        if not self._enabled or self._container_client is None:
            return 0
        count = 0
        try:
            for blob in self._container_client.list_blobs():
                try:
                    self._container_client.delete_blob(blob.name)
                    count += 1
                except Exception:
                    pass
        except Exception as exc:
            logger.error("Failed to list blobs for deletion: %s", exc)
        return count

    def list_models(self):
        """Return list of blob names."""
        if not self._enabled or self._container_client is None:
            return []
        try:
            return [b.name for b in self._container_client.list_blobs()]
        except Exception:
            return []

    @property
    def enabled(self) -> bool:
        return self._enabled


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
blob_model_store = BlobModelStore()
