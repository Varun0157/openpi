#!/usr/bin/env python3
"""Debug the create_rlds_dataset function specifically."""

import sys
import traceback
import time

def debug_create_rlds_dataset():
    print("=== Debugging create_rlds_dataset function ===")
    
    try:
        print("1. Importing modules...")
        import openpi.training.config as _config
        import openpi.training.data_loader as _data_loader
        print("   ✓ Modules imported")
        
        print("2. Getting config...")
        config = _config.get_config("pi0_fast_droid_finetune")
        data_config = config.data.create(config.assets_dirs, config.model)
        print(f"   ✓ Config loaded")
        print(f"   - action_horizon: {config.model.action_horizon}")
        print(f"   - batch_size: {config.batch_size}")
        print(f"   - rlds_data_dir: {data_config.rlds_data_dir}")
        print(f"   - action_space: {data_config.action_space}")
        
        print("3. Calling create_rlds_dataset...")
        dataset = _data_loader.create_rlds_dataset(
            data_config, 
            config.model.action_horizon, 
            config.batch_size, 
            shuffle=False
        )
        print(f"   ✓ create_rlds_dataset returned: {type(dataset)}")
        print(f"   - Dataset length: {len(dataset)}")
        
        print("4. Inspecting the dataset object...")
        print(f"   - Dataset attributes: {[attr for attr in dir(dataset) if not attr.startswith('_')]}")
        
        # Look at the actual DroidRldsDataset parameters
        if hasattr(dataset, 'batch_size'):
            print(f"   - Internal batch_size: {dataset.batch_size}")
        if hasattr(dataset, 'shuffle'):
            print(f"   - Internal shuffle: {dataset.shuffle}")
            
        print("5. Testing iteration on create_rlds_dataset result...")
        iterator = iter(dataset)
        
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Timeout in create_rlds_dataset iteration")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        start_time = time.time()
        batch = next(iterator)
        elapsed = time.time() - start_time
        signal.alarm(0)
        
        print(f"   ✓ SUCCESS in {elapsed:.2f}s")
        print(f"   - Batch keys: {list(batch.keys())}")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"     {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"     {key}: dict with keys {list(value.keys())}")
        
        print("6. Comparing with direct DroidRldsDataset...")
        from openpi.training.droid_rlds_dataset import DroidRldsDataset
        
        direct_dataset = DroidRldsDataset(
            data_dir=data_config.rlds_data_dir,
            batch_size=config.batch_size,
            shuffle=False,
            action_chunk_size=config.model.action_horizon,
            action_space=data_config.action_space,
        )
        
        print(f"   - Direct dataset type: {type(direct_dataset)}")
        print(f"   - Same type? {type(dataset) == type(direct_dataset)}")
        
        # Test direct dataset
        direct_iterator = iter(direct_dataset)
        signal.alarm(15)
        direct_batch = next(direct_iterator)
        signal.alarm(0)
        print(f"   ✓ Direct dataset also works")
        
        print("=== create_rlds_dataset works fine! ===")
        return True
        
    except TimeoutError as e:
        print(f"   ✗ TIMEOUT: {e}")
        return False
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_create_rlds_dataset()
    sys.exit(0 if success else 1)