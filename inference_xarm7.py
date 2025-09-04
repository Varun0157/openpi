import dataclasses
import os
import time

import cv2
import numpy as np
from PIL import Image
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class InferenceConfig:
    checkpoint_dir: str = "checkpoints/pi0_fast_droid_finetune_low_mem/my_experiment/499"

    use_gui: bool = False  # Set to False for headless mode on server
    control_frequency: float = 15.0  # Hz, matching DROID data collection frequency

    image_width: int = 320
    image_height: int = 180
    camera_position: list = dataclasses.field(
        default_factory=lambda: [
            0.09378594165842033,
            0.4828175119051615,
            0.19511362660974355,
        ]
    )
    camera_orientation_euler: list = dataclasses.field(
        default_factory=lambda: [-1.859357113506073, -8.922049171955493e-05, -2.557306600133795]
    )

    urdf_path: str = "./Embodiment-Codes-RRC/URDF/src_xarm/airobot/urdfs/xarm7_robot.urdf"
    end_effector_link_index: int = 7

    max_timesteps: int = 1000
    action_horizon: int = 16

    save_images: bool = True
    image_output_dir: str = "./inference_images"


class XArm7InferenceEnv:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.setup_pybullet()
        self.setup_robot()
        self.setup_camera()
        self.setup_image_saving()

    def setup_pybullet(self):
        """Initialize PyBullet simulation."""
        if self.config.use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)  # Headless mode

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0 / 240.0)  # High frequency simulation

    def setup_robot(self):
        """Load the xARM7 robot URDF."""
        self.robot_id = p.loadURDF(self.config.urdf_path, [0, 0, 0], useFixedBase=True)

        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = list(range(self.num_joints))

        print(f"Loaded robot with {self.num_joints} joints")

        initial_state = [
            0.11024115004068978,
            -0.29933845557606764,
            0.05442884930953815,
            0.9136438149155726,
            0.08774621794390562,
            1.2080655976492474,
            0.058147608614682836,
        ]
        for i, angle in enumerate(initial_state):
            if i >= self.num_joints:
                break
            p.resetJointState(self.robot_id, i, angle)

    def setup_camera(self):
        """Setup camera parameters for image capture."""
        self.camera_position = np.array(self.config.camera_position)
        self.camera_orientation = p.getQuaternionFromEuler(self.config.camera_orientation_euler)

        self.projection_matrix = self._compute_projection_matrix()

    def update_intrinsic_matrix(self, k, old_dims, new_dims):
        old_h, old_w = old_dims
        new_h, new_w = new_dims
        scale_w, scale_h = new_w / old_w, new_h / old_h

        k_new = k.copy()
        k_new[0, 0] *= scale_w
        k_new[1, 1] *= scale_h
        k_new[0, 2] *= scale_w
        k_new[1, 2] *= scale_h

        return k_new

    def _compute_projection_matrix(self):
        """Convert camera intrinsics to PyBullet projection matrix."""
        near, far = 0.1, 3.1
        w, h = self.config.image_width, self.config.image_height

        self.camera_intrinsics = np.array(
            [[522.6506958007812, 0.0, 639.2378540039062], [0.0, 522.6506958007812, 352.5005798339844], [0.0, 0.0, 1.0]]
        )
        self.camera_intrinsics = self.update_intrinsic_matrix(self.camera_intrinsics, (720, 1280), (h, w))

        fx = self.camera_intrinsics[0, 0]
        fy = self.camera_intrinsics[1, 1]
        cx = self.camera_intrinsics[0, 2]
        cy = self.camera_intrinsics[1, 2]

        a = (near + far) / (near - far)
        b = 2 * near * far / (near - far)

        projection_matrix = [
            [2 * fx / w, 0, (w - 2 * cx) / w, 0],
            [0, 2 * fy / h, (2 * cy - h) / h, 0],
            [0, 0, a, b],
            [0, 0, -1, 0],
        ]

        return np.array(projection_matrix).T.reshape(16).tolist()

    def setup_image_saving(self):
        """Setup directory for saving images."""
        if self.config.save_images:
            os.makedirs(self.config.image_output_dir, exist_ok=True)

    def capture_image(self, step_idx=None):
        """Capture RGB image from the camera."""
        # Compute camera target position
        rot_matrix = R.from_quat(self.camera_orientation).as_matrix()
        camera_target = self.camera_position + rot_matrix @ np.array([0, 0, 1])

        # Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position, cameraTargetPosition=camera_target, cameraUpVector=[0, 0, 1]
        )

        # Capture image
        width, height = self.config.image_width, self.config.image_height
        _, _, rgb_img, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        # Convert to numpy array and remove alpha channel
        rgb_array = np.reshape(rgb_img, (height, width, 4))[:, :, :3]

        # Save image if requested
        if self.config.save_images and step_idx is not None:
            image_path = os.path.join(self.config.image_output_dir, f"step_{step_idx:04d}.png")
            Image.fromarray(rgb_array).save(image_path)

        return rgb_array

    def get_robot_state(self):
        """Get current robot joint positions."""
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_positions = [state[0] for state in joint_states]
        return np.array(joint_positions)

    def execute_action(self, action):
        """Execute action on the robot (joint position control)."""
        # The policy outputs 8-dimensional actions (7 joint deltas + 1 gripper absolute)
        # For xARM7, we use all 7 joint dimensions (joint deltas)
        if len(action) >= 7:
            joint_deltas = action[:7]  # Take first 7 dimensions for joint deltas
        else:
            raise Exception(f"Action has insufficient dimensions: {len(action)} < 7")

        # Get current joint positions
        current_joint_positions = self.get_robot_state()

        # Convert delta actions to absolute positions
        # First 7 dimensions are deltas, add to current positions
        target_joint_positions = current_joint_positions[: len(joint_deltas)] + joint_deltas

        # Ensure we only control the available joints
        target_joint_positions = target_joint_positions[: self.num_joints]
        joint_indices_to_control = self.joint_indices[: len(target_joint_positions)]

        # Set joint position targets
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=joint_indices_to_control,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_joint_positions,
        )

        # Step simulation
        steps_per_action = int(240.0 / self.config.control_frequency)
        for _ in range(steps_per_action):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)


def create_policy_input(image, robot_state, prompt):
    """Create input dictionary for the policy."""
    # Resize image to expected size (224x224 for DROID policy)
    image_resized = cv2.resize(image, (224, 224))
    state = np.concatenate([robot_state, [0.0]])[:8] if len(robot_state) < 8 else robot_state[:8]

    return {
        "observation/image": image_resized,
        "observation/state": state,
        "prompt": prompt,
    }


def main():
    config = InferenceConfig(
        checkpoint_dir="checkpoints/pi0_fast_droid_finetune_low_mem/my_experiment/499",
        use_gui=False,  # Set to False for headless server mode
        save_images=True,
    )

    print("Setting up inference environment...")

    # Setup environment
    env = XArm7InferenceEnv(config)

    # Load policy
    print("Loading policy...")
    try:
        train_config = _config.get_config("pi0_fast_droid_finetune_low_mem")
        policy = _policy_config.create_trained_policy(train_config, config.checkpoint_dir)
        print("Policy loaded successfully!")
    except Exception as e:
        print(f"Error loading policy: {e}")
        print("Please ensure your checkpoint directory path is correct.")
        p.disconnect()
        return

    # Inference loop
    print("Starting inference loop...")
    prompt = "Move object into or out of container (ex: drawer, clothes hamper, plate, trashcan, washer)"

    # Track action execution
    actions_from_chunk = []
    actions_executed = 0

    try:
        for step in range(config.max_timesteps):
            print(f"Step {step + 1}/{config.max_timesteps}")

            # Capture current observation
            image = env.capture_image(step)
            robot_state = env.get_robot_state()

            # Get new action prediction if needed
            if actions_executed == 0 or actions_executed >= config.action_horizon:
                print("  Getting new action prediction...")

                # Create policy input
                policy_input = create_policy_input(image, robot_state, prompt)

                # Run inference
                result = policy.infer(policy_input)
                actions_from_chunk = result["actions"]  # Shape: [action_horizon, action_dim]
                actions_executed = 0

                print(f"  Predicted {len(actions_from_chunk)} actions")

            # Execute next action from chunk
            action = actions_from_chunk[actions_executed]
            print(f"  Executing action: {action}")

            env.execute_action(action)
            actions_executed += 1

            # Small delay to match control frequency
            time.sleep(1.0 / config.control_frequency)

    except KeyboardInterrupt:
        print("\nInference interrupted by user.")
    except Exception as e:
        print(f"\nError during inference: {e}")
    finally:
        print("Cleaning up...")
        p.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
