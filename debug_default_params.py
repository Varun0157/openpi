#!/usr/bin/env python3
"""Debug the default parameters issue in DroidRldsDataset."""

import sys
import traceback
import time

def test_with_different_params():
    print("=== Testing DroidRldsDataset with different parameters ===")
    
    from openpi.training.droid_rlds_dataset import DroidRldsDataset, DroidActionSpace
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Timeout")
    signal.signal(signal.SIGALRM, timeout_handler)
    
    def test_dataset(name, **kwargs):
        print(f"\n{name}:")
        try:
            dataset = DroidRldsDataset(
                data_dir="data",
                batch_size=256,
                shuffle=False,
                action_chunk_size=16,
                action_space=DroidActionSpace.JOINT_POSITION,
                **kwargs
            )
            
            iterator = iter(dataset)
            signal.alarm(20)  # 20 second timeout
            start_time = time.time()
            batch = next(iterator)
            elapsed = time.time() - start_time
            signal.alarm(0)
            
            print(f"   ✓ SUCCESS in {elapsed:.2f}s")
            return True
            
        except TimeoutError:
            print(f"   ✗ TIMEOUT")
            return False
        except Exception as e:
            print(f"   ✗ ERROR: {e}")
            return False
    
    # Test 1: Our working parameters (small values)
    success1 = test_dataset(
        "1. Small parameters (known working)",
        max_loaded_steps_per_episode=10,
        shuffle_buffer_size=100,
    )
    
    # Test 2: Default parameters (what create_rlds_dataset uses)
    success2 = test_dataset(
        "2. Default parameters (what create_rlds_dataset uses)",
        # These are the defaults from DroidRldsDataset.__init__
        max_loaded_steps_per_episode=100,
        shuffle_buffer_size=250_000,
    )
    
    # Test 3: Larger max_loaded_steps_per_episode but small buffer
    success3 = test_dataset(
        "3. Large max_loaded_steps_per_episode, small buffer",
        max_loaded_steps_per_episode=100,
        shuffle_buffer_size=100,
    )
    
    # Test 4: Small max_loaded_steps_per_episode but large buffer
    success4 = test_dataset(
        "4. Small max_loaded_steps_per_episode, large buffer",
        max_loaded_steps_per_episode=10,
        shuffle_buffer_size=250_000,
    )
    
    print(f"\n=== Results ===")
    print(f"Small params: {'✓' if success1 else '✗'}")
    print(f"Default params: {'✓' if success2 else '✗'}")
    print(f"Large steps, small buffer: {'✓' if success3 else '✗'}")
    print(f"Small steps, large buffer: {'✓' if success4 else '✗'}")
    
    if not success2:
        print(f"\n*** FOUND THE ISSUE: Default parameters cause timeout! ***")
        if success3 and not success4:
            print("*** The problem is the large shuffle_buffer_size=250,000 ***")
        elif success4 and not success3:
            print("*** The problem is max_loaded_steps_per_episode=100 ***")
        else:
            print("*** Both parameters contribute to the issue ***")

if __name__ == "__main__":
    test_with_different_params()