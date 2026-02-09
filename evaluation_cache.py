"""
Evaluation cache for meta-SR.

Caches operator bundle evaluation results to avoid redundant SR runs.
Uses SQLite for persistent storage, similar to completions_cache.py.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import Column, String, Float, Text, Boolean, DateTime, create_engine, select
from sqlalchemy.orm import Session

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class EvaluationCacheEntry(Base):
    """SQLite table for caching evaluation results."""
    __tablename__ = "evaluations"

    # Primary key: hash of full request
    request_hash = Column(String, primary_key=True)
    # Indexed columns for easy lookup/invalidation
    bundle_hash = Column(String, index=True)
    dataset_name = Column(String, index=True)
    eval_type = Column(String)  # 'quick' or 'full'
    # Result data
    score = Column(Float)
    traces_json = Column(Text)
    error = Column(Text, nullable=True)
    timed_out = Column(Boolean, default=False)
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class PySRCacheEntry(Base):
    """SQLite table for caching PySR evaluation results."""
    __tablename__ = "pysr_evaluations"

    # Primary key: hash of full request
    request_hash = Column(String, primary_key=True)
    # Indexed columns for easy lookup/invalidation
    config_hash = Column(String, index=True)
    dataset_name = Column(String, index=True)
    # Result data
    r2_score = Column(Float)
    best_equation = Column(Text, nullable=True)
    best_loss = Column(Float)
    error = Column(Text, nullable=True)
    timed_out = Column(Boolean, default=False)
    runtime_seconds = Column(Float, default=0.0)
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class EvaluationCacheDB:
    """SQLite cache for evaluation results."""

    def __init__(self, database_path: str = "caches/evaluation_cache.db"):
        db_dir = os.path.dirname(database_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{database_path}")
        Base.metadata.create_all(self.engine)

    def _make_bundle_hash(self, bundle_codes: Dict[str, str]) -> str:
        """Create a deterministic hash for a bundle from its operator codes."""
        # Canonical representation: sorted keys
        key_data = {
            "selection": bundle_codes.get("selection", ""),
            "mutation": bundle_codes.get("mutation", ""),
            "crossover": bundle_codes.get("crossover", ""),
            "fitness": bundle_codes.get("fitness", ""),
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _make_cache_key(
        self,
        bundle_codes: Dict[str, str],
        dataset_name: str,
        seed: int,
        data_seed: int,
        max_samples: Optional[int],
        run_index: int,
        sr_kwargs: Dict,
        target_noise: float = 0.0,
    ) -> str:
        """Create a deterministic hash key for the evaluation request."""
        bundle_hash = self._make_bundle_hash(bundle_codes)
        key_data = {
            "bundle_hash": bundle_hash,
            "dataset_name": dataset_name,
            "seed": seed,
            "data_seed": data_seed,
            "max_samples": max_samples,
            "run_index": run_index,
            "sr_kwargs": sr_kwargs,
            "target_noise": target_noise,
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def lookup(
        self,
        bundle_codes: Dict[str, str],
        dataset_name: str,
        seed: int,
        data_seed: int,
        max_samples: Optional[int],
        run_index: int,
        sr_kwargs: Dict,
        target_noise: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Look up a cached evaluation result. Returns None if not found."""
        request_hash = self._make_cache_key(
            bundle_codes, dataset_name, seed, data_seed, max_samples, run_index, sr_kwargs,
            target_noise
        )

        stmt = select(
            EvaluationCacheEntry.score,
            EvaluationCacheEntry.traces_json,
            EvaluationCacheEntry.error,
            EvaluationCacheEntry.timed_out,
        ).where(EvaluationCacheEntry.request_hash == request_hash)

        with Session(self.engine) as session:
            result = session.execute(stmt).first()
            if result:
                return {
                    "score": result[0],
                    "traces": json.loads(result[1]) if result[1] else [],
                    "error": result[2],
                    "timed_out": result[3],
                }
        return None

    def store(
        self,
        bundle_codes: Dict[str, str],
        dataset_name: str,
        seed: int,
        data_seed: int,
        max_samples: Optional[int],
        run_index: int,
        sr_kwargs: Dict,
        target_noise: float = 0.0,
        score: float = 0.0,
        traces: List[str] = None,
        error: Optional[str] = None,
        timed_out: bool = False,
        eval_type: str = "full",
    ) -> None:
        """Store an evaluation result in the cache."""
        if traces is None:
            traces = []
        bundle_hash = self._make_bundle_hash(bundle_codes)
        request_hash = self._make_cache_key(
            bundle_codes, dataset_name, seed, data_seed, max_samples, run_index, sr_kwargs,
            target_noise
        )

        entry = EvaluationCacheEntry(
            request_hash=request_hash,
            bundle_hash=bundle_hash,
            dataset_name=dataset_name,
            eval_type=eval_type,
            score=score,
            traces_json=json.dumps(traces),
            error=error,
            timed_out=timed_out,
            created_at=datetime.utcnow(),
        )

        with Session(self.engine) as session:
            session.merge(entry)  # merge handles insert-or-update
            session.commit()

    def get_bundle_hash(self, bundle_codes: Dict[str, str]) -> str:
        """Public method to get bundle hash for external use."""
        return self._make_bundle_hash(bundle_codes)


# Global cache instance
_cache: Optional[EvaluationCacheDB] = None
_cache_enabled: bool = True


def get_cache() -> Optional[EvaluationCacheDB]:
    """Get the global cache instance, creating it if needed."""
    global _cache
    if not _cache_enabled:
        return None
    if _cache is None:
        _cache = EvaluationCacheDB()
    return _cache


def set_cache_path(database_path: str):
    """Change the cache database path. Creates a new cache instance."""
    global _cache
    _cache = EvaluationCacheDB(database_path)


def disable_cache():
    """Disable the evaluation cache."""
    global _cache_enabled
    _cache_enabled = False


def enable_cache():
    """Enable the evaluation cache."""
    global _cache_enabled
    _cache_enabled = True


def is_cache_enabled() -> bool:
    """Check if the cache is enabled."""
    return _cache_enabled


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the cache."""
    cache = get_cache()
    if cache is None:
        return {"enabled": False, "num_entries": 0}
    with Session(cache.engine) as session:
        count = session.query(EvaluationCacheEntry).count()
        return {"enabled": True, "num_entries": count}


def clear_cache():
    """Clear all entries from the cache."""
    cache = get_cache()
    if cache is None:
        return
    with Session(cache.engine) as session:
        session.query(EvaluationCacheEntry).delete()
        session.commit()


# =============================================================================
# PySR Evaluation Cache
# =============================================================================


class PySRCacheDB:
    """SQLite cache for PySR evaluation results."""

    def __init__(self, database_path: str = "caches/pysr_evaluation_cache.db"):
        db_dir = os.path.dirname(database_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{database_path}")
        Base.metadata.create_all(self.engine)

    def _make_config_hash(
        self,
        mutation_weights: Dict[str, float],
        pysr_kwargs: Dict,
        custom_mutation_code: Optional[Dict[str, str]],
        allow_custom_mutations: bool,
    ) -> str:
        """Create a deterministic hash for a PySR configuration."""
        key_data = {
            "mutation_weights": mutation_weights,
            "pysr_kwargs": pysr_kwargs,
            "custom_mutation_code": custom_mutation_code,
            "allow_custom_mutations": allow_custom_mutations,
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _make_cache_key(
        self,
        mutation_weights: Dict[str, float],
        pysr_kwargs: Dict,
        dataset_name: str,
        seed: int,
        data_seed: int,
        max_samples: Optional[int],
        run_index: int,
        custom_mutation_code: Optional[Dict[str, str]],
        allow_custom_mutations: bool,
        pysr_model_kwargs: Optional[Dict] = None,
        target_noise: float = 0.0,
    ) -> str:
        """Create a deterministic hash key for the PySR evaluation request."""
        config_hash = self._make_config_hash(
            mutation_weights, pysr_kwargs, custom_mutation_code, allow_custom_mutations
        )
        key_data = {
            "config_hash": config_hash,
            "dataset_name": dataset_name,
            "seed": seed,
            "data_seed": data_seed,
            "max_samples": max_samples,
            "run_index": run_index,
            "pysr_model_kwargs": pysr_model_kwargs,
            "target_noise": target_noise,
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def lookup(
        self,
        mutation_weights: Dict[str, float],
        pysr_kwargs: Dict,
        dataset_name: str,
        seed: int,
        data_seed: int,
        max_samples: Optional[int],
        run_index: int,
        custom_mutation_code: Optional[Dict[str, str]] = None,
        allow_custom_mutations: bool = False,
        pysr_model_kwargs: Optional[Dict] = None,
        target_noise: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Look up a cached PySR evaluation result. Returns None if not found."""
        request_hash = self._make_cache_key(
            mutation_weights, pysr_kwargs, dataset_name, seed, data_seed,
            max_samples, run_index, custom_mutation_code, allow_custom_mutations,
            pysr_model_kwargs, target_noise
        )

        stmt = select(
            PySRCacheEntry.r2_score,
            PySRCacheEntry.best_equation,
            PySRCacheEntry.best_loss,
            PySRCacheEntry.error,
            PySRCacheEntry.timed_out,
            PySRCacheEntry.runtime_seconds,
        ).where(PySRCacheEntry.request_hash == request_hash)

        with Session(self.engine) as session:
            result = session.execute(stmt).first()
            if result:
                return {
                    "r2_score": result[0],
                    "best_equation": result[1],
                    "best_loss": result[2],
                    "error": result[3],
                    "timed_out": result[4],
                    "runtime_seconds": result[5],
                }
        return None

    def store(
        self,
        mutation_weights: Dict[str, float],
        pysr_kwargs: Dict,
        dataset_name: str,
        seed: int,
        data_seed: int,
        max_samples: Optional[int],
        run_index: int,
        custom_mutation_code: Optional[Dict[str, str]],
        allow_custom_mutations: bool,
        pysr_model_kwargs: Optional[Dict] = None,
        target_noise: float = 0.0,
        r2_score: float = 0.0,
        best_equation: Optional[str] = None,
        best_loss: float = float("inf"),
        error: Optional[str] = None,
        timed_out: bool = False,
        runtime_seconds: float = 0.0,
    ) -> None:
        """Store a PySR evaluation result in the cache."""
        config_hash = self._make_config_hash(
            mutation_weights, pysr_kwargs, custom_mutation_code, allow_custom_mutations
        )
        request_hash = self._make_cache_key(
            mutation_weights, pysr_kwargs, dataset_name, seed, data_seed,
            max_samples, run_index, custom_mutation_code, allow_custom_mutations,
            pysr_model_kwargs, target_noise
        )

        entry = PySRCacheEntry(
            request_hash=request_hash,
            config_hash=config_hash,
            dataset_name=dataset_name,
            r2_score=r2_score,
            best_equation=best_equation,
            best_loss=best_loss,
            error=error,
            timed_out=timed_out,
            runtime_seconds=runtime_seconds,
            created_at=datetime.utcnow(),
        )

        with Session(self.engine) as session:
            session.merge(entry)
            session.commit()

    def get_config_hash(
        self,
        mutation_weights: Dict[str, float],
        pysr_kwargs: Dict,
        custom_mutation_code: Optional[Dict[str, str]] = None,
        allow_custom_mutations: bool = False,
    ) -> str:
        """Public method to get config hash for external use."""
        return self._make_config_hash(
            mutation_weights, pysr_kwargs, custom_mutation_code, allow_custom_mutations
        )


# Global PySR cache instance
_pysr_cache: Optional[PySRCacheDB] = None
_pysr_cache_enabled: bool = True


def get_pysr_cache() -> Optional[PySRCacheDB]:
    """Get the global PySR cache instance, creating it if needed."""
    global _pysr_cache
    if not _pysr_cache_enabled:
        return None
    if _pysr_cache is None:
        _pysr_cache = PySRCacheDB()
    return _pysr_cache


def set_pysr_cache_path(database_path: str):
    """Change the PySR cache database path. Creates a new cache instance."""
    global _pysr_cache
    _pysr_cache = PySRCacheDB(database_path)


def disable_pysr_cache():
    """Disable the PySR evaluation cache."""
    global _pysr_cache_enabled
    _pysr_cache_enabled = False


def enable_pysr_cache():
    """Enable the PySR evaluation cache."""
    global _pysr_cache_enabled
    _pysr_cache_enabled = True


def is_pysr_cache_enabled() -> bool:
    """Check if the PySR cache is enabled."""
    return _pysr_cache_enabled


def get_pysr_cache_stats() -> Dict[str, Any]:
    """Get statistics about the PySR cache."""
    cache = get_pysr_cache()
    if cache is None:
        return {"enabled": False, "num_entries": 0}
    with Session(cache.engine) as session:
        count = session.query(PySRCacheEntry).count()
        return {"enabled": True, "num_entries": count}


def clear_pysr_cache():
    """Clear all entries from the PySR cache."""
    cache = get_pysr_cache()
    if cache is None:
        return
    with Session(cache.engine) as session:
        session.query(PySRCacheEntry).delete()
        session.commit()
