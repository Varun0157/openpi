#!/usr/bin/env python3
"""Debug script to understand the DroidRldsDataset loading issue."""

import os
import sys
import traceback
import time

def debug_droid_dataset():
    print("=== Debugging DroidRldsDataset directly ===")
    
    try:
        print("1. Setting up basic imports...")
        from openpi.training.droid_rlds_dataset import DroidRldsDataset, DroidActionSpace
        print("   ✓ Imports successful")
        
        print("2. Checking data directory...")
        data_dir = "data"
        droid_path = os.path.join(data_dir, "droid")
        version_path = os.path.join(droid_path, "1.0.0")
        
        print(f"   - data_dir: {data_dir}")
        print(f"   - droid_path exists: {os.path.exists(droid_path)}")
        print(f"   - version_path exists: {os.path.exists(version_path)}")
        
        if os.path.exists(version_path):
            files = os.listdir(version_path)
            print(f"   - files in version dir: {files[:5]}...")  # Show first 5 files
            
        print("3. Testing tensorflow datasets builder...")
        import tensorflow_datasets as tfds
        
        try:
            print("   - Attempting to create tfds builder...")
            builder = tfds.builder("droid", data_dir=data_dir)
            print(f"   ✓ Builder created: {builder}")
            print(f"   - Builder info: {builder.info}")
        except Exception as e:
            print(f"   ✗ Builder creation failed: {e}")
            print("   - This might be the root cause!")
            return False
            
        print("4. Testing DroidRldsDataset creation...")
        try:
            dataset = DroidRldsDataset(
                data_dir=data_dir,
                batch_size=4,  # Small batch size for testing
                shuffle=False,  # No shuffle for easier debugging
                action_chunk_size=16,
                action_space=DroidActionSpace.JOINT_POSITION,
                max_loaded_steps_per_episode=10,  # Very small for testing
                shuffle_buffer_size=100,  # Very small buffer
            )
            print("   ✓ DroidRldsDataset created successfully")
        except Exception as e:
            print(f"   ✗ DroidRldsDataset creation failed: {e}")
            traceback.print_exc()
            return False
            
        print("5. Testing dataset iteration...")
        try:
            print("   - Creating iterator...")
            iterator = iter(dataset)
            print("   - Getting first batch with timeout...")
            
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("Timeout getting first batch")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
            
            first_batch = next(iterator)
            signal.alarm(0)
            
            print("   ✓ First batch retrieved successfully!")
            print(f"   - Batch keys: {list(first_batch.keys())}")
            
        except TimeoutError:
            print("   ✗ Timeout occurred - dataset iteration is hanging")
            return False
        except Exception as e:
            print(f"   ✗ Error during iteration: {e}")
            traceback.print_exc()
            return False
            
        print("=== Debug completed successfully ===")
        return True
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_droid_dataset()
    sys.exit(0 if success else 1)