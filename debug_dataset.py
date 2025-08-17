#!/usr/bin/env python3
"""Debug script to understand why the DROID dataset iteration is stuck."""

import sys
import traceback
import time

# Add some debug prints to understand the flow
def debug_dataset_loading():
    print("=== Starting dataset debug ===")
    
    try:
        print("1. Importing modules...")
        import openpi.training.config as _config
        import openpi.training.data_loader as _data_loader
        print("   ✓ Modules imported successfully")
        
        print("2. Getting config...")
        config = _config.get_config("pi0_fast_droid_finetune")
        print(f"   ✓ Config loaded: {config.data}")
        
        print("3. Creating data config...")
        data_config = config.data.create(config.assets_dirs, config.model)
        print(f"   ✓ Data config created")
        print(f"   - repo_id: {data_config.repo_id}")
        print(f"   - rlds_data_dir: {data_config.rlds_data_dir}")
        print(f"   - action_space: {data_config.action_space}")
        
        print("4. Creating RLDS dataset...")
        dataset = _data_loader.create_rlds_dataset(
            data_config, 
            config.model.action_horizon, 
            config.batch_size, 
            shuffle=False
        )
        print(f"   ✓ RLDS dataset created: {type(dataset)}")
        print(f"   - Dataset length: {len(dataset)}")
        
        print("5. Testing dataset iteration...")
        iterator = iter(dataset)
        print("   ✓ Iterator created")
        
        print("6. Attempting to get first batch...")
        start_time = time.time()
        try:
            # Set a timeout to avoid hanging
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("Dataset iteration timed out after 30 seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
            
            first_batch = next(iterator)
            signal.alarm(0)  # Cancel the alarm
            
            elapsed = time.time() - start_time
            print(f"   ✓ First batch retrieved in {elapsed:.2f}s")
            print(f"   - Batch keys: {list(first_batch.keys())}")
            print(f"   - Batch shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in first_batch.items()]}")
            
        except TimeoutError as e:
            print(f"   ✗ {e}")
            return False
        except Exception as e:
            print(f"   ✗ Error getting first batch: {e}")
            traceback.print_exc()
            return False
            
        print("7. Testing a few more batches...")
        for i in range(3):
            try:
                signal.alarm(10)  # 10 second timeout for subsequent batches
                batch = next(iterator)
                signal.alarm(0)
                print(f"   ✓ Batch {i+2} retrieved")
            except Exception as e:
                print(f"   ✗ Error getting batch {i+2}: {e}")
                break
                
        print("=== Dataset debug completed successfully ===")
        return True
        
    except Exception as e:
        print(f"✗ Error during dataset debug: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_dataset_loading()
    sys.exit(0 if success else 1)