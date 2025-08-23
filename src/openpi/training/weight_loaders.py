import dataclasses
import pdb
import logging
import re
import time
import tracemalloc
import threading
import os
from typing import Protocol, runtime_checkable, List, Tuple
from functools import wraps

import flax.traverse_util
import numpy as np
import matplotlib.pyplot as plt

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Memory profiler that tracks memory usage over time during function execution."""

    def __init__(self, func_name: str, sample_interval: float = 0.01):
        self.func_name = func_name
        self.sample_interval = sample_interval
        self.memory_data: List[Tuple[float, float, float]] = []  # (timestamp, rss_mb, tracemalloc_mb)
        self.start_time = None
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process() if psutil else None

    def _monitor_memory(self):
        """Background thread function to monitor memory usage."""
        while self.monitoring:
            current_time = time.time() - self.start_time

            # Get RSS memory from psutil if available
            rss_mb = 0.0
            if self.process:
                try:
                    rss_mb = self.process.memory_info().rss / 1024 / 1024
                except:
                    pass

            # Get tracemalloc memory
            tracemalloc_mb = 0.0
            if tracemalloc.is_tracing():
                try:
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc_mb = current / 1024 / 1024
                except:
                    pass

            self.memory_data.append((current_time, rss_mb, tracemalloc_mb))
            time.sleep(self.sample_interval)

    def start_monitoring(self):
        """Start memory monitoring."""
        self.start_time = time.time()
        self.memory_data.clear()
        self.monitoring = True

        # Start tracemalloc if not already started
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop memory monitoring and save results."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        self._save_results()

    def _save_results(self):
        """Save memory profiling results to file and create graph."""
        if not self.memory_data:
            logger.warning(f"No memory data collected for {self.func_name}")
            return

        # Create output directory
        output_dir = "memory_profiles"
        os.makedirs(output_dir, exist_ok=True)

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.func_name}_{timestamp_str}"

        # Save raw data
        data_file = os.path.join(output_dir, f"{base_filename}.txt")
        with open(data_file, "w") as f:
            f.write("# Memory profiling data for function: {}\n".format(self.func_name))
            f.write("# Columns: time_seconds, rss_memory_mb, tracemalloc_memory_mb\n")
            for time_s, rss_mb, trace_mb in self.memory_data:
                f.write(f"{time_s:.6f}\t{rss_mb:.2f}\t{trace_mb:.2f}\n")

        # Create graph
        self._create_graph(output_dir, base_filename)

        # Log summary
        max_rss = max(data[1] for data in self.memory_data) if any(data[1] > 0 for data in self.memory_data) else 0
        max_trace = max(data[2] for data in self.memory_data) if any(data[2] > 0 for data in self.memory_data) else 0
        duration = self.memory_data[-1][0] if self.memory_data else 0

        logger.info(f"Memory profile for {self.func_name} saved:")
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info(f"  Peak RSS memory: {max_rss:.2f} MB")
        logger.info(f"  Peak tracemalloc memory: {max_trace:.2f} MB")
        logger.info(f"  Data file: {data_file}")
        logger.info(f"  Graph: {os.path.join(output_dir, base_filename)}.png")

    def _create_graph(self, output_dir: str, base_filename: str):
        """Create memory usage graph."""
        try:
            times = [data[0] for data in self.memory_data]
            rss_memory = [data[1] for data in self.memory_data]
            trace_memory = [data[2] for data in self.memory_data]

            plt.figure(figsize=(12, 8))

            # Plot RSS memory if available
            if any(rss > 0 for rss in rss_memory):
                plt.subplot(2, 1, 1)
                plt.plot(times, rss_memory, "b-", linewidth=2, label="RSS Memory")
                plt.xlabel("Time (seconds)")
                plt.ylabel("Memory (MB)")
                plt.title(f"RSS Memory Usage - {self.func_name}")
                plt.grid(True, alpha=0.3)
                plt.legend()

            # Plot tracemalloc memory if available
            if any(trace > 0 for trace in trace_memory):
                plt.subplot(2, 1, 2)
                plt.plot(times, trace_memory, "r-", linewidth=2, label="Tracemalloc Memory")
                plt.xlabel("Time (seconds)")
                plt.ylabel("Memory (MB)")
                plt.title(f"Tracemalloc Memory Usage - {self.func_name}")
                plt.grid(True, alpha=0.3)
                plt.legend()

            plt.tight_layout()
            graph_file = os.path.join(output_dir, f"{base_filename}.png")
            plt.savefig(graph_file, dpi=150, bbox_inches="tight")
            plt.close()

        except Exception as e:
            logger.error(f"Failed to create memory usage graph: {e}")


def memory_profile(func_name: str = None, sample_interval: float = 0.01):
    """Decorator to profile memory usage of a function.

    Args:
        func_name: Custom name for the function (defaults to actual function name)
        sample_interval: How often to sample memory usage in seconds
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            profiler = MemoryProfiler(name, sample_interval)

            profiler.start_monitoring()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.stop_monitoring()

        return wrapper

    return decorator


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        logger.info("inside load method")

        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        # pdb.set_trace()
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)

        # pdb.set_trace()
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@memory_profile("merge_params", sample_interval=0.005)
def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            # result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v
            result[k] = v.astype(flat_ref[k].dtype)

    # flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
