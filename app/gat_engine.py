"""
Internal GAT Engine
Provides in-process GAT processing, replacing the previous HTTP microservice
approach.  All GAT classes live in the app.gat package and are instantiated
directly here; no network calls are made.
"""

import logging
import os
import time
import random
from types import SimpleNamespace
from typing import Any, Dict, Optional

from app.layer3_models import GATProcessingRequest, GATProcessingResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import real GAT components; fall back to simulation if unavailable
# (e.g. when torch / torch-geometric are not installed in the environment)
# ---------------------------------------------------------------------------
_GAT_AVAILABLE = False
_SiameseGATNetwork = None
_GATInferenceEngine = None
_PyTorchDataConverter = None

try:
    from app.gat.gat_network import SiameseGATNetwork, GATInferenceEngine  # type: ignore[import]
    from app.gat.data_processor import PyTorchDataConverter

    _SiameseGATNetwork = SiameseGATNetwork
    _GATInferenceEngine = GATInferenceEngine
    _PyTorchDataConverter = PyTorchDataConverter
    _GAT_AVAILABLE = True
    logger.info("app.gat components imported successfully")
except ImportError as exc:
    logger.warning("app.gat components not available (%s) – running in simulation mode", exc)


# ---------------------------------------------------------------------------
# Internal GAT Engine
# ---------------------------------------------------------------------------

class InternalGATEngine:
    """
    In-process GAT engine.

    On first use it tries to load the trained model from the standard checkpoint
    path.  If PyTorch / torch-geometric are not installed it falls back to a
    lightweight simulation that preserves the same response contract.
    """

    def __init__(self):
        self._model = None
        self._inference_engine = None
        self._pytorch_converter = None
        self._output_dim = 64
        self._device = "cpu"
        self._initialized = False

    # ------------------------------------------------------------------
    # Lazy initialisation – called once on first use
    # ------------------------------------------------------------------

    def _ensure_initialized(self):
        if self._initialized:
            return
        self._initialized = True

        if not _GAT_AVAILABLE:
            logger.info("GAT engine running in simulation mode (PyTorch unavailable)")
            return

        try:
            gat_config = {
                "input_dim": 56,
                "hidden_dim": 128,
                "output_dim": 64,
                "num_heads": 8,
                "dropout": 0.1,
                "temporal_dim": 8,
            }
            self._output_dim = gat_config["output_dim"]
            self._model = _SiameseGATNetwork(gat_config)
            self._inference_engine = _GATInferenceEngine(self._model, self._device)
            self._pytorch_converter = _PyTorchDataConverter()

            # Look for pre-trained weights in the gat-service checkpoints or
            # alongside the app.gat package itself.
            _repo_root = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            candidates = [
                os.path.join(_repo_root, "gat-service", "models", "gat_model.pth"),
                os.path.join(_repo_root, "checkpoints", "gat_model.pth"),
            ]
            model_path = next((p for p in candidates if os.path.exists(p)), None)

            if model_path:
                import torch  # type: ignore[import]

                state = torch.load(model_path, map_location=self._device)
                self._model.load_state_dict(state)
                logger.info("Loaded GAT model weights from %s", model_path)
            else:
                logger.info("No pre-trained GAT weights found – using random initialisation")

        except Exception as exc:
            logger.error(
                "Failed to initialise GAT model: %s – falling back to simulation", exc
            )
            self._model = None
            self._inference_engine = None
            self._pytorch_converter = None

    # ------------------------------------------------------------------
    # Public interface (mirrors GATCloudInterface.process_temporal_graph)
    # ------------------------------------------------------------------

    def process_request(self, request: GATProcessingRequest) -> GATProcessingResponse:
        """
        Process a temporal graph request and return GAT embeddings / scores.

        Uses real PyTorch model when available, otherwise returns simulation output.
        """
        self._ensure_initialized()

        if self._inference_engine is not None and self._pytorch_converter is not None:
            return self._real_inference(request)
        return self._simulate(request)

    # ------------------------------------------------------------------
    # Real PyTorch inference
    # ------------------------------------------------------------------

    def _real_inference(self, request: GATProcessingRequest) -> GATProcessingResponse:
        try:
            # Build a duck-typed graph object that PyTorchDataConverter accepts.
            graph = request.graph
            nodes_ns = [
                SimpleNamespace(
                    node_id=n.node_id,
                    timestamp=n.timestamp,
                    event_type=n.event_type,
                    behavioral_vector=n.behavioral_vector,
                    signature=n.signature,
                    nonce=n.nonce,
                )
                for n in graph.nodes
            ]
            edges_ns = [
                SimpleNamespace(
                    source_node_id=e.source_node_id,
                    target_node_id=e.target_node_id,
                    time_delta=e.time_delta,
                    event_transition=e.event_transition,
                )
                for e in graph.edges
            ]
            graph_ns = SimpleNamespace(
                session_id=graph.session_id,
                user_id=graph.user_id,
                nodes=nodes_ns,
                edges=edges_ns,
                window_start=graph.window_start,
                window_end=graph.window_end,
                total_events=graph.total_events,
                session_duration=graph.session_duration,
                event_diversity=graph.event_diversity,
                avg_time_between_events=graph.avg_time_between_events,
            )

            graph_data_dict = self._pytorch_converter.convert_to_pytorch(graph_ns)

            data = SimpleNamespace(
                x=graph_data_dict["x"],
                edge_index=graph_data_dict["edge_index"],
                temporal_features=graph_data_dict["temporal_features"],
                batch=graph_data_dict["batch"],
            )

            profile_vector = request.user_profile_vector or [0.0] * self._output_dim
            result = self._inference_engine.authenticate(
                data,
                profile_vector,
                request.similarity_threshold,
            )

            return GATProcessingResponse(
                session_vector=result["session_vector"],
                similarity_score=result.get("similarity_score"),
                processing_time_ms=result.get("processing_time_ms", 0.0),
            )

        except Exception as exc:
            logger.error("Real GAT inference failed (%s) – falling back to simulation", exc)
            return self._simulate(request)

    # ------------------------------------------------------------------
    # Simulation fallback
    # ------------------------------------------------------------------

    def _simulate(self, request: GATProcessingRequest) -> GATProcessingResponse:
        start = time.time()

        session_vector = [random.uniform(-1.0, 1.0) for _ in range(self._output_dim)]
        similarity = (
            random.uniform(0.70, 0.95)
            if request.user_profile_vector
            else random.uniform(0.50, 0.80)
        )
        processing_time_ms = (time.time() - start) * 1000.0

        return GATProcessingResponse(
            session_vector=session_vector,
            similarity_score=similarity,
            processing_time_ms=processing_time_ms,
        )


# Module-level singleton
_internal_engine: Optional[InternalGATEngine] = None


def get_internal_engine() -> InternalGATEngine:
    """Return (or lazily create) the module-level singleton."""
    global _internal_engine
    if _internal_engine is None:
        _internal_engine = InternalGATEngine()
    return _internal_engine
