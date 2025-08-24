"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import logging
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

# Configure logging
logging.basicConfig(level=logging.INFO)


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=8,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    logging.info(f"Creating RLDS dataset with data_dir: {data_config.rlds_data_dir}")
    logging.info(f"Action horizon: {action_horizon}, batch_size: {batch_size}")
    
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    logging.info("Base RLDS dataset created")
    
    logging.info("Applying transforms...")
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    logging.info("Transforms applied")
    
    logging.info(f"Dataset length: {len(dataset)}")
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        logging.info(f"Using max_frames={max_frames}, num_batches={num_batches}")
    else:
        num_batches = len(dataset) // batch_size
        logging.info(f"Using full dataset, num_batches={num_batches}")
    
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    logging.info("RLDSDataLoader created")
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    logging.info(f"Starting compute_norm_stats for config: {config_name}")
    
    logging.info("Loading config...")
    config = _config.get_config(config_name)
    logging.info(f"Config loaded: {config.name}")
    
    logging.info("Creating data config...")
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"Data config created. RLDS data dir: {data_config.rlds_data_dir}")

    if data_config.rlds_data_dir is not None:
        logging.info("Creating RLDS dataloader...")
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
        logging.info(f"RLDS dataloader created. Num batches: {num_batches}")
    else:
        logging.info("Creating Torch dataloader...")
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, max_frames
        )
        logging.info(f"Torch dataloader created. Num batches: {num_batches}")

    keys = ["state", "actions"]
    logging.info(f"Initializing stats for keys: {keys}")
    stats = {key: normalize.RunningStats() for key in keys}

    logging.info(f"Starting iteration over {num_batches} batches...")
    for batch_idx, batch in enumerate(tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats")):
        if batch_idx == 0:
            logging.info(f"First batch received! Keys: {list(batch.keys())}")
            for key in keys:
                if key in batch:
                    logging.info(f"Key '{key}' shape: {np.asarray(batch[key]).shape}")
                else:
                    logging.error(f"Key '{key}' not found in batch!")
        
        for key in keys:
            if key in batch:
                values = np.asarray(batch[key][0])
                stats[key].update(values.reshape(-1, values.shape[-1]))
            else:
                logging.error(f"Missing key '{key}' in batch {batch_idx}")
        
        if batch_idx % 100 == 0 and batch_idx > 0:
            logging.info(f"Processed {batch_idx} batches")

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
