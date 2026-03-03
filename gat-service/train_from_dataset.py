"""Train GAT model from dataset JSON."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class GraphData:
    x: "torch.Tensor"  # type: ignore[name-defined]
    edge_index: "torch.Tensor"  # type: ignore[name-defined]
    temporal_features: "torch.Tensor"  # type: ignore[name-defined]
    batch: "torch.Tensor | None"  # type: ignore[name-defined]


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_graph(session: dict, torch_module):
    events = session["events"]
    x = torch_module.tensor([event["node_vector"] for event in events], dtype=torch_module.float32)
    timestamps = torch_module.tensor([[event["timestamp"] - session["window_start"]] for event in events], dtype=torch_module.float32)

    edge_sources = []
    edge_targets = []
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            edge_sources.append(i)
            edge_targets.append(j)
            break

    if not edge_sources:
        edge_sources = [0]
        edge_targets = [0]

    edge_index = torch_module.tensor([edge_sources, edge_targets], dtype=torch_module.long)
    return GraphData(x=x, edge_index=edge_index, temporal_features=timestamps, batch=None)


def add_noise(x, torch_module, scale: float = 0.01):
    noise = torch_module.randn_like(x) * scale
    return x + noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="../datasets/fast_dataset.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checkpoint", default="checkpoints/gat_checkpoint.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    dataset = load_dataset(dataset_path)
    sessions = dataset.get("sessions", [])
    if not sessions:
        raise SystemExit("No sessions found in dataset")

    if args.dry_run:
        print(f"Loaded {len(sessions)} session(s) from {dataset_path}")
        print("Dry-run mode: no training executed")
        return

    try:
        import torch  # type: ignore[reportMissingImports]
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Torch is required for training: {exc}")

    try:
        from tqdm import tqdm  # type: ignore[reportMissingImports]
    except Exception:  # pragma: no cover
        tqdm = None

    from gat_network import SiameseGATNetwork, GATTrainer

    gat_config = {
        "input_dim": dataset.get("node_vector_dim", 60),
        "hidden_dim": 64,
        "output_dim": 64,
        "num_heads": 8,
        "dropout": 0.1,
        "temporal_dim": 16,
    }

    model = SiameseGATNetwork(gat_config)
    trainer = GATTrainer(model=model, learning_rate=0.001, device="cpu")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0))
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    session = sessions[0]
    base_graph = build_graph(session, torch)

    total_epochs = args.epochs
    epoch_iter = range(start_epoch, total_epochs)
    progress = tqdm(epoch_iter, desc="Training", unit="epoch") if tqdm is not None else None

    try:
        for epoch in (progress or epoch_iter):
            anchor = base_graph
            positive = GraphData(
                x=add_noise(base_graph.x, torch),
                edge_index=base_graph.edge_index,
                temporal_features=base_graph.temporal_features,
                batch=None,
            )
            negative = GraphData(
                x=add_noise(base_graph.x, torch, scale=0.2),
                edge_index=base_graph.edge_index,
                temporal_features=base_graph.temporal_features,
                batch=None,
            )
            loss = trainer.train_batch(anchor, positive, negative)

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": trainer.optimizer.state_dict(),
                    "loss": loss,
                    "timestamp": time.time(),
                },
                checkpoint_path,
            )

            if progress is None:
                print(f"Epoch {epoch + 1}/{total_epochs} loss: {loss:.6f}")
            else:
                progress.set_postfix(loss=f"{loss:.6f}")
    except KeyboardInterrupt:
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": trainer.optimizer.state_dict(),
                "loss": loss,
                "timestamp": time.time(),
            },
            checkpoint_path,
        )
        print(f"Interrupted. Checkpoint saved to {checkpoint_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
