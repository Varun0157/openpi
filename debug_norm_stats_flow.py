#!/usr/bin/env python3
"""Debug the exact flow used in compute_norm_stats.py to find where it hangs."""

import sys
import traceback
import time
import numpy as np

def debug_norm_stats_flow():
    print("=== Debugging compute_norm_stats.py flow ===")
    
    try:
        print("1. Importing required modules...")
        import openpi.training.config as _config
        import openpi.training.data_loader as _data_loader
        import openpi.transforms as transforms
        print("   ✓ Modules imported")
        
        print("2. Creating RemoveStrings transform...")
        class RemoveStrings(transforms.DataTransformFn):
            def __call__(self, x: dict) -> dict:
                result = {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}
                print(f"   RemoveStrings: input keys {list(x.keys())} -> output keys {list(result.keys())}")
                return result
        print("   ✓ RemoveStrings transform created")
        
        print("3. Getting config...")
        config = _config.get_config("pi0_fast_droid_finetune")
        data_config = config.data.create(config.assets_dirs, config.model)
        print(f"   ✓ Config loaded - batch_size: {config.batch_size}")
        
        print("4. Creating RLDS dataset (same as before)...")
        dataset = _data_loader.create_rlds_dataset(
            data_config, config.model.action_horizon, config.batch_size, shuffle=False
        )
        print(f"   ✓ Raw dataset created, length: {len(dataset)}")
        
        print("5. Testing raw dataset iteration...")
        raw_iterator = iter(dataset)
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Timeout")
        signal.signal(signal.SIGALRM, timeout_handler)
        
        signal.alarm(10)
        raw_batch = next(raw_iterator)
        signal.alarm(0)
        print(f"   ✓ Raw batch retrieved - keys: {list(raw_batch.keys())}")
        
        print("6. Creating IterableTransformedDataset...")
        transformed_dataset = _data_loader.IterableTransformedDataset(
            dataset,
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                RemoveStrings(),
            ],
            is_batched=True,
        )
        print(f"   ✓ Transformed dataset created")
        print(f"   - Repack transforms: {len(data_config.repack_transforms.inputs)}")
        print(f"   - Data transforms: {len(data_config.data_transforms.inputs)}")
        
        print("7. Testing transformed dataset iteration...")
        transformed_iterator = iter(transformed_dataset)
        
        signal.alarm(30)  # Longer timeout for transforms
        transformed_batch = next(transformed_iterator)
        signal.alarm(0)
        print(f"   ✓ Transformed batch retrieved - keys: {list(transformed_batch.keys())}")
        
        print("8. Computing num_batches...")
        max_frames = None
        if max_frames is not None and max_frames < len(dataset):
            num_batches = max_frames // config.batch_size
        else:
            num_batches = len(dataset) // config.batch_size
        print(f"   ✓ num_batches: {num_batches}")
        
        print("9. Creating RLDSDataLoader...")
        data_loader = _data_loader.RLDSDataLoader(
            transformed_dataset,
            num_batches=num_batches,
        )
        print(f"   ✓ RLDSDataLoader created")
        
        print("10. Testing final data loader iteration...")
        final_iterator = iter(data_loader)
        
        signal.alarm(30)
        final_batch = next(final_iterator)
        signal.alarm(0)
        print(f"   ✓ Final batch retrieved - keys: {list(final_batch.keys())}")
        
        print("11. Testing the exact loop from compute_norm_stats.py...")
        keys = ["state", "actions"]
        
        # Check what keys are actually available
        print(f"   - Available keys in batch: {list(final_batch.keys())}")
        for key in keys:
            if key in final_batch:
                print(f"   - {key}: shape {np.asarray(final_batch[key][0]).shape}")
            else:
                print(f"   - {key}: NOT FOUND")
                
        print("=== All tests passed! ===")
        return True
        
    except TimeoutError:
        print("   ✗ Timeout occurred - found the hanging point!")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_norm_stats_flow()
    sys.exit(0 if success else 1)