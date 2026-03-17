"""
app.azure — Azure cloud service wrappers

Provides unified access to all Azure-backed storage, logging, and
enrollment services used by the CBSA backend.
"""
from app.azure.behavioral_logger import behavioral_logger, BehavioralFileLogger, BEHAVIORAL_LOG_DIR
from app.azure.blob_model_store import blob_model_store, BlobModelStore
from app.azure.cosmos_logger import cosmos_logger, CosmosComputationLogger
from app.azure.cosmos_profile_store import cosmos_profile_store, CosmosProfileStore
from app.azure.enrollment_store import enrollment_store, EnrollmentStore, ENROLLMENT_DURATION_SECONDS

__all__ = [
    "behavioral_logger", "BehavioralFileLogger", "BEHAVIORAL_LOG_DIR",
    "blob_model_store", "BlobModelStore",
    "cosmos_logger", "CosmosComputationLogger",
    "cosmos_profile_store", "CosmosProfileStore",
    "enrollment_store", "EnrollmentStore", "ENROLLMENT_DURATION_SECONDS",
]
