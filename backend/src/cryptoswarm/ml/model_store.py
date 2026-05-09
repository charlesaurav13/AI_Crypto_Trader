"""ModelStore — save and load ML model artifacts to/from disk."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelStore:
    def __init__(self, model_dir: str = "models") -> None:
        self._dir = Path(model_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, obj: object) -> None:
        path = self._dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.debug("ModelStore: saved %s to %s", name, path)

    def load(self, name: str) -> object | None:
        path = self._dir / f"{name}.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.debug("ModelStore: loaded %s from %s", name, path)
        return obj

    def exists(self, name: str) -> bool:
        return (self._dir / f"{name}.pkl").exists()
