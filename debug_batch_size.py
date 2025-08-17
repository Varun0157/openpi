#!/usr/bin/env python3
"""Debug if the issue is related to batch size."""

import sys
import traceback
import time

def test_batch_size(batch_size):
    print(f"\n=== Testing batch size {batch_size} ===")
    
    try:
        import openpi.training.config as _config
        import openpi.training.data_loader as _data_loader
        from openpi.training.droid_rlds_dataset import DroidRldsDataset, DroidActionSpace
        
        print(f"1. Creating DroidRldsDataset with batch_size={batch_size}...")
        dataset = DroidRldsDataset(
            data_dir="data",
            batch_size=batch_size,
            shuffle=False,
            action_chunk_size=16,
            action_space=DroidActionSpace.JOINT_POSITION,
            max_loaded_steps_per_episode=10,  # Small for testing
            shuffle_buffer_size=100,  # Small buffer
        )
        print(f"   ✓ Dataset created")
        
        print(f"2. Testing iteration...")
        iterator = iter(dataset)
        
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Timeout with batch_size={batch_size}")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)  # 15 second timeout
        
        start_time = time.time()
        batch = next(iterator)
        elapsed = time.time() - start_time
        signal.alarm(0)
        
        print(f"   ✓ SUCCESS in {elapsed:.2f}s - batch keys: {list(batch.keys())}")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"     {key}: {value.shape}")
        return True
        
    except TimeoutError as e:
        print(f"   ✗ TIMEOUT: {e}")
        return False
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        traceback.print_exc()
        return False

def main():
    print("=== Debugging batch size issues ===")
    
    # Test progressively larger batch sizes
    batch_sizes = [1, 4, 16, 64, 128, 256]
    
    for batch_size in batch_sizes:
        success = test_batch_size(batch_size)
        if not success:
            print(f"\n*** ISSUE FOUND: batch_size={batch_size} fails ***")
            print(f"*** Last working batch_size: {batch_sizes[batch_sizes.index(batch_size)-1] if batch_sizes.index(batch_size) > 0 else 'None'} ***")
            break
    else:
        print("\n*** All batch sizes work - issue is elsewhere ***")

if __name__ == "__main__":
    main()