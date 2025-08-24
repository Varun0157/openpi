"""
RLDS-based data loader for DROID.
While openpi typically uses LeRobot's data loader, it is not currently scalable enough for larger datasets like DROID.
Thus, we provide a data loader example here that uses the RLDS data format.
The data loader also applies a few DROID-specific data filters / transformations.
"""

from enum import Enum
from enum import auto

import gc
import logging


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()


class DroidRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # We default to joint position actions, since they allow policy evaluation in simulation.
        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 2_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
    ):
        print("=== DroidRldsDataset.__init__ STARTED ===")
        logging.info("=== DroidRldsDataset.__init__ STARTED ===")
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        print("Importing dlimp...")
        logging.info("Importing dlimp...")
        import dlimp as dl
        print("Importing tensorflow...")
        logging.info("Importing tensorflow...")
        import tensorflow as tf
        print("Importing tensorflow_datasets...")
        logging.info("Importing tensorflow_datasets...")
        import tensorflow_datasets as tfds
        print("All imports complete")
        logging.info("All imports complete")

        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        print("Configuring TensorFlow...")
        logging.info("Configuring TensorFlow...")
        tf.config.set_visible_devices([], "GPU")
        print("TensorFlow configured")
        logging.info("TensorFlow configured")

        # # attempted fix
        # tf.config.threading.set_inter_op_parallelism_threads(2)
        # tf.config.threading.set_intra_op_parallelism_threads(4)

        # gc.collect()

        print(f"Loading DROID dataset from: {data_dir}")
        logging.info(f"Loading DROID dataset from: {data_dir}")
        print("Creating TFDS builder...")
        builder = tfds.builder("droid", data_dir=data_dir)
        print("TFDS builder created")
        logging.info(f"TFDS builder created")
        print(f"TFDS builder info: {builder.info}")
        logging.info(f"TFDS builder info: {builder.info}")
        
        print("Creating DLataset from RLDS...")
        dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)
        print("DLataset created successfully")
        logging.info("Successfully created DLataset from RLDS")

        # Filter out any unsuccessful trajectories -- we use the file name to check this
        print("Applying success filter to trajectories")
        logging.info("Applying success filter to trajectories")
        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
            )
        )
        print("Success filter applied")
        logging.info("Success filter applied")

        # Repeat dataset so we never run out of data.
        dataset = dataset.repeat()

        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            # Important: we use joint *position* action space -- easier to simulate!
            
            print("=== RESTRUCTURE FUNCTION CALLED ===")
            
            # Log trajectory keys for debugging
            print(f"Trajectory keys: {list(traj.keys())}")
            if "action" in traj:
                print(f"Action shape: {tf.shape(traj['action'])}")
            else:
                print("ERROR: No 'action' key in trajectory!")
                
            if "observation" in traj:
                print(f"Observation keys: {list(traj['observation'].keys())}")
            else:
                print("ERROR: No 'observation' key in trajectory!")

            print("Creating actions with extra zero column...")
            # Create actions from traj["action"] with extra column of zeros
            actions = tf.concat(
                [traj["action"], tf.zeros_like(traj["action"][..., :1])],
                axis=-1,
            )
            print("Actions created successfully")

            print("Getting exterior image...")
            exterior_img = traj["observation"]["image"]
            print("Creating wrist image...")
            wrist_img = tf.zeros_like(exterior_img)
            print("Getting instruction...")
            instruction = traj["language_instruction"]
            
            print("Returning restructured trajectory...")
            return {
                "actions": actions,
                "observation": {
                    "image": exterior_img,
                    "wrist_image": wrist_img,
                    "state": traj["observation"]["state"],
                },
                "prompt": instruction,
            }

        print("Applying restructure transform")
        logging.info("Applying restructure transform")
        dataset = dataset.traj_map(restructure, num_parallel_calls)
        print("Restructure transform applied")
        logging.info("Restructure transform applied")

        def chunk_actions(traj):
            """Splits episode into action chunks."""
            print("=== CHUNK_ACTIONS FUNCTION CALLED ===")
            traj_len = tf.shape(traj["actions"])[0]
            print(f"Trajectory length: {traj_len}")

            # For each step in the trajectory, construct indices for the next n actions
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )

            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            print("Gathering action chunks...")
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            print("Action chunking complete")
            return traj

        print("Applying action chunking")
        logging.info("Applying action chunking")
        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)
        print("Action chunking applied")
        logging.info("Action chunking applied")

        # Flatten: map from trajectory dataset to dataset of individual action chunks
        print("Flattening dataset")
        logging.info("Flattening dataset")
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)
        print("Dataset flattened")
        logging.info("Dataset flattened")

        # Filter out frames where actions are idle. Must be done after flattening, as filter should apply per-frame.
        def filter_idle(traj):
            """Filter out chunks with idle actions.
            --> we filter if at least first half of chunk does not move.
            """
            if action_space == DroidActionSpace.JOINT_POSITION:
                # Compute delta to first position in action chunk
                return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2] - traj["actions"][:1]) > 1e-3)
            return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2]) > 1e-3)

        logging.info("Applying idle action filter")
        dataset = dataset.filter(filter_idle)
        logging.info("Idle action filter applied")

        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            return traj

        logging.info("Applying image decoding")
        dataset = dataset.frame_map(decode_images, num_parallel_calls)
        logging.info("Image decoding applied")

        # Shuffle, batch
        logging.info(f"Shuffling dataset with buffer size: {shuffle_buffer_size}")
        dataset = dataset.shuffle(shuffle_buffer_size)
        logging.info(f"Batching dataset with batch size: {batch_size}")
        dataset = dataset.batch(batch_size)
        logging.info("Dataset preparation complete")

        # Note =>> Seems to reduce memory usage without affecting speed?
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # gc.collect()

    def __iter__(self):
        logging.info("Starting dataset iteration")
        try:
            count = 0
            for batch in self.dataset.as_numpy_iterator():
                if count == 0:
                    logging.info(f"First batch keys: {list(batch.keys())}")
                    if "actions" in batch:
                        logging.info(f"First batch actions shape: {batch['actions'].shape}")
                    if "observation" in batch:
                        logging.info(f"First batch observation keys: {list(batch['observation'].keys())}")
                        if "state" in batch["observation"]:
                            logging.info(f"First batch state shape: {batch['observation']['state'].shape}")
                count += 1
                if count % 100 == 0:
                    logging.info(f"Processed {count} batches")
                yield batch
        except Exception as e:
            logging.error(f"Error during iteration: {e}")
            logging.error(f"Error type: {type(e)}")
            raise

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        # WARNING: This is a hardcoded estimate and may not reflect actual data availability!
        logging.warning("Using hardcoded dataset length of 32,000 - this may not match actual data!")
        return 32_000
