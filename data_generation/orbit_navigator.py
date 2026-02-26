"""
Orbital Navigation Data Collector for AirSim.

Drives a simulated drone in circular orbits around objects of interest in an
Unreal Engine 4 environment, capturing RGB, segmentation, and depth images
from multiple camera viewpoints. Generates bounding box annotations from
segmentation masks.

Refactored from NavAble orbit_bb.py for the GuideTWSI project.
"""

import setup_path  # noqa: F401 — must be imported before airsim
import json
import cv2
import airsim
import os
import numpy as np
import sys
import math
import time
import argparse
from typing import Optional

import airsim_utils


# Default segmentation color-to-label mapping.
# Override via --color-map-yaml for different environments.
DEFAULT_COLOR_MAP = {
    (153, 108, 6): "tactile block",
    (112, 105, 191): "pedestrian sign",
    (89, 121, 72): "tactile block",
}


class Position:
    """Wraps AirSim position values."""

    def __init__(self, pos) -> None:
        self.x: float = pos.x_val
        self.y: float = pos.y_val
        self.z: float = pos.z_val


class OrbitNavigator:
    """Controls a drone to orbit a center point and capture multi-modal data.

    Args:
        radius: Orbit radius in meters.
        altitude: Orbit altitude in positive meters above ground.
        speed: Orbit speed in meters/second.
        iterations: Number of full 360-degree orbits.
        center: [x, y] direction vector pointing to orbit center.
        snapshots: Number of snapshots to capture per orbit.
        environment_name: Name of the UE4 environment.
        run_number: Run identifier for output directory naming.
        segmentation_objects: Dict mapping object names to segmentation IDs.
            If None, all scene objects get auto-assigned IDs.
        color_to_object_map: Dict mapping RGB tuples to label strings.
    """

    def __init__(
        self,
        radius: float = 2,
        altitude: float = 0,
        speed: float = 2,
        iterations: int = 1,
        center: list[float] = [1, 0],
        snapshots: int = 100,
        environment_name: str = "Test",
        run_number: int = 1,
        segmentation_objects: Optional[dict[str, int]] = None,
        color_to_object_map: Optional[dict[tuple, str]] = None,
    ) -> None:
        self.radius = radius
        self.altitude = altitude
        self.speed = speed
        self.iterations = iterations
        self.snapshots = snapshots
        self.snapshot_delta: Optional[float] = None
        self.next_snapshot: Optional[float] = None
        self.z: Optional[float] = None
        self.snapshot_index: int = 0
        self.takeoff: bool = False
        self.environment_name = environment_name
        self.run_number = run_number

        if self.snapshots is not None and self.snapshots > 0:
            self.snapshot_delta = 360 / self.snapshots

        if self.iterations <= 0:
            self.iterations = 1

        if len(center) != 2:
            raise ValueError("Expecting '[x,y]' for the center direction vector")

        cx = float(center[0])
        cy = float(center[1])
        length = math.sqrt((cx * cx) + (cy * cy))
        if length == 0:
            cx, cy = 0, 0
        else:
            cx /= length
            cy /= length
        cx *= self.radius
        cy *= self.radius

        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        # Wait for drone position to stabilize
        self.home = self.client.getMultirotorState().kinematics_estimated.position
        start = time.time()
        count = 0
        while count < 100:
            pos = self.client.getMultirotorState().kinematics_estimated.position
            if abs(pos.z_val - self.home.z_val) > 1:
                count = 0
                self.home = pos
                if time.time() - start > 10:
                    print("Drone position is drifting, waiting for it to settle...")
                    start = time.time()
            else:
                count += 1

        self.center = pos
        self.center.x_val += cx
        self.center.y_val += cy

        # Set segmentation IDs for scene objects
        all_objects = self.client.simListSceneObjects()
        if segmentation_objects:
            for obj_name, seg_id in segmentation_objects.items():
                if obj_name in all_objects:
                    self.client.simSetSegmentationObjectID(obj_name, seg_id, False)
                    print(f"Set segmentation ID: {obj_name} -> {seg_id}")
        else:
            # Auto-assign IDs to all objects starting from 4
            i = 4
            for obj_name in all_objects:
                self.client.simSetSegmentationObjectID(obj_name, i, False)
                i += 1
                if i == 256:
                    i = 4

        self.color_to_object_map = color_to_object_map or DEFAULT_COLOR_MAP

    def start(self) -> None:
        """Arms the drone, takes off, executes orbital flight, and lands."""
        print("Arming the drone...")
        self.client.armDisarm(True)

        start = self.client.getMultirotorState().kinematics_estimated.position
        landed = self.client.getMultirotorState().landed_state
        if not self.takeoff and landed == airsim.LandedState.Landed:
            self.takeoff = True
            print("Taking off...")
            self.client.takeoffAsync().join()
            start = self.client.getMultirotorState().kinematics_estimated.position
            z = -self.altitude
        else:
            print(f"Already flying, orbiting at current altitude {start.z_val}")
            z = start.z_val

        print(f"Climbing to position: {start.x_val},{start.y_val},{z}")
        self.client.moveToPositionAsync(start.x_val, start.y_val, z, self.speed).join()
        self.z = z

        print("Ramping up to speed...")
        count = 0
        self.start_angle: Optional[float] = None
        self.next_snapshot = None

        ramptime = self.radius / 10
        self.start_time = time.time()

        while count < self.iterations:
            if self.snapshots > 0 and not (self.snapshot_index < self.snapshots):
                break
            now = time.time()
            speed = self.speed
            diff = now - self.start_time
            if diff < ramptime:
                speed = self.speed * diff / ramptime
            elif ramptime > 0:
                print("Reached full speed...")
                ramptime = 0

            lookahead_angle = speed / self.radius

            pos = self.client.getMultirotorState().kinematics_estimated.position
            dx = pos.x_val - self.center.x_val
            dy = pos.y_val - self.center.y_val
            angle_to_center = math.atan2(dy, dx)
            camera_heading = (angle_to_center - math.pi) * 180 / math.pi

            lookahead_x = self.center.x_val + self.radius * math.cos(
                angle_to_center + lookahead_angle
            )
            lookahead_y = self.center.y_val + self.radius * math.sin(
                angle_to_center + lookahead_angle
            )

            vx = lookahead_x - pos.x_val
            vy = lookahead_y - pos.y_val

            if self._track_orbits(angle_to_center * 180 / math.pi):
                count += 1
                print(f"Completed {count} orbits")

            self.camera_heading = camera_heading
            self.client.moveByVelocityZAsync(
                vx,
                vy,
                z,
                1,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(False, camera_heading),
            )

        self.client.moveToPositionAsync(start.x_val, start.y_val, z, 2).join()

        if self.takeoff:
            if z < self.home.z_val:
                print("Descending")
            print("Landing...")
            self.client.landAsync().join()
            print("Disarming.")
            self.client.armDisarm(False)

    def _track_orbits(self, angle: float) -> bool:
        """Tracks angular progress and triggers snapshots."""
        if angle < 0:
            angle += 360

        if self.start_angle is None:
            self.start_angle = angle
            if self.snapshot_delta:
                self.next_snapshot = angle + self.snapshot_delta
            self.previous_angle = angle
            self.shifted = False
            self.previous_sign = None
            self.previous_diff = None
            self.quarter = False
            return False

        if self.previous_angle is None:
            self.previous_angle = angle
            return False

        if self.previous_angle > 350 and angle < 10:
            if self.snapshot_delta and self.next_snapshot >= 360:
                self.next_snapshot -= 360
            return False

        self.previous_angle = angle

        if self.snapshot_delta and angle > self.next_snapshot:
            print(f"Taking snapshot {self.snapshot_index}")
            self._take_snapshot()
            self.next_snapshot += self.snapshot_delta

        diff = abs(angle - self.start_angle)
        if diff > 45:
            self.quarter = True

        crossing = False
        if self.quarter and self.previous_diff is not None and diff != self.previous_diff:
            direction = self._sign(self.previous_diff - diff)
            if self.previous_sign is None:
                self.previous_sign = direction
            elif self.previous_sign > 0 and direction < 0:
                if diff < 45:
                    self.quarter = False
                    if self.snapshots <= self.snapshot_index + 1:
                        crossing = True
            self.previous_sign = direction
        self.previous_diff = diff

        return crossing

    def _take_snapshot(self) -> None:
        """Captures RGB, segmentation, and depth images from front and top-down cameras."""
        run_dir, environment_number = self._create_directories(
            self.environment_name, self.run_number
        )

        # Hold position for stable capture
        pos = self.client.getMultirotorState().kinematics_estimated.position
        self.client.moveToPositionAsync(
            pos.x_val,
            pos.y_val,
            self.z,
            0.5,
            10,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(False, self.camera_heading),
        ).join()

        image_type_rgb = airsim.ImageType.Scene
        image_type_seg = airsim.ImageType.Segmentation
        image_type_depth = airsim.ImageType.DepthPlanar

        # Front-view camera
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", image_type_rgb),
            airsim.ImageRequest("0", image_type_seg, False, False),
            airsim.ImageRequest("0", image_type_depth, True, False),
        ])

        # Top-down camera
        bottom_responses = self.client.simGetImages([
            airsim.ImageRequest("3", image_type_rgb),
            airsim.ImageRequest("3", image_type_seg, False, False),
            airsim.ImageRequest("3", image_type_depth, True, False),
        ])

        image_number = self.snapshot_index

        # Process front-view images
        rgb_image = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        rgb_image = cv2.imdecode(rgb_image, cv2.IMREAD_COLOR)

        seg_image = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
        seg_image = seg_image.reshape(responses[1].height, responses[1].width, 3)

        depth_img = np.array(responses[2].image_data_float, dtype=np.float32)
        depth_array = depth_img.reshape(responses[2].height, responses[2].width)
        airsim_utils.write_pfm("depth_image.pfm", depth_array)
        depth_normalized = (depth_array - depth_array.min()) / (
            depth_array.max() - depth_array.min()
        ) * 255
        depth_img = depth_normalized.astype(np.uint8)

        # Process top-down images
        rgb_td_image = np.frombuffer(bottom_responses[0].image_data_uint8, dtype=np.uint8)
        rgb_td_image = cv2.imdecode(rgb_td_image, cv2.IMREAD_COLOR)

        seg_td_image = np.frombuffer(bottom_responses[1].image_data_uint8, dtype=np.uint8)
        seg_td_image = seg_td_image.reshape(
            bottom_responses[1].height, bottom_responses[1].width, 3
        )

        depth_td_img = np.array(bottom_responses[2].image_data_float, dtype=np.float32)
        depth_td_array = depth_td_img.reshape(
            bottom_responses[2].height, bottom_responses[2].width
        )
        airsim_utils.write_pfm("depth_td_image.pfm", depth_td_array)
        depth_td_normalized = (depth_td_array - depth_td_array.min()) / (
            depth_td_array.max() - depth_td_array.min()
        ) * 255
        depth_td_img = depth_td_normalized.astype(np.uint8)

        # Save raw images
        prefix = f"{self.environment_name}_{environment_number}_{self.run_number}_{image_number}"
        cv2.imwrite(os.path.join(run_dir, "rgb", f"{prefix}_FV_rgb.png"), rgb_image)
        cv2.imwrite(os.path.join(run_dir, "seg", f"{prefix}_FV_segmentation.png"), seg_image)
        cv2.imwrite(os.path.join(run_dir, "depth", f"{prefix}_FV_depth.png"), depth_img)
        cv2.imwrite(os.path.join(run_dir, "rgb", f"{prefix}_TD_rgb.png"), rgb_td_image)
        cv2.imwrite(os.path.join(run_dir, "seg", f"{prefix}_TD_segmentation.png"), seg_td_image)
        cv2.imwrite(os.path.join(run_dir, "depth", f"{prefix}_TD_depth.png"), depth_td_img)

        # Extract and save bounding boxes
        bounding_boxes = self._get_bounding_boxes(seg_image, rgb_image)
        td_bounding_boxes = self._get_bounding_boxes(seg_td_image, rgb_td_image)

        # Draw bounding boxes on copies for visualization
        for bbox in bounding_boxes:
            x_min = bbox["bounding_box"]["x_min"]
            y_min = bbox["bounding_box"]["y_min"]
            x_max = bbox["bounding_box"]["x_max"]
            y_max = bbox["bounding_box"]["y_max"]
            cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                rgb_image,
                bbox["object_name"],
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )

        for bbox in td_bounding_boxes:
            x_min = bbox["bounding_box"]["x_min"]
            y_min = bbox["bounding_box"]["y_min"]
            x_max = bbox["bounding_box"]["x_max"]
            y_max = bbox["bounding_box"]["y_max"]
            cv2.rectangle(rgb_td_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                rgb_td_image,
                bbox["object_name"],
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )

        cv2.imwrite(os.path.join(run_dir, "rgb_bb", f"{prefix}_FV_rgb_bb.png"), rgb_image)
        cv2.imwrite(os.path.join(run_dir, "rgb_bb", f"{prefix}_TD_rgb_bb.png"), rgb_td_image)

        with open(os.path.join(run_dir, "label", f"{prefix}_FV.json"), "w") as f:
            json.dump(bounding_boxes, f)

        with open(os.path.join(run_dir, "label", f"{prefix}_TD.json"), "w") as f:
            json.dump(td_bounding_boxes, f)

        self.snapshot_index += 1
        self.start_time = time.time()

    def _get_bounding_boxes(
        self, segmentation_image: np.ndarray, rgb_image: np.ndarray
    ) -> list[dict]:
        """Extracts bounding boxes from a segmentation image by color matching."""
        seg_resized = cv2.resize(
            segmentation_image, (rgb_image.shape[1], rgb_image.shape[0])
        )
        seg_resized = cv2.cvtColor(seg_resized, cv2.COLOR_BGR2RGB)

        bounding_boxes = []
        for target_color, object_name in self.color_to_object_map.items():
            mask = np.all(seg_resized == np.array(target_color), axis=-1)
            non_zero_coords = np.argwhere(mask)

            if non_zero_coords.size > 0:
                y_min, x_min = np.min(non_zero_coords, axis=0)
                y_max, x_max = np.max(non_zero_coords, axis=0)
                bounding_boxes.append({
                    "object_name": object_name,
                    "object_id": target_color,
                    "bounding_box": {
                        "x_min": int(x_min),
                        "y_min": int(y_min),
                        "x_max": int(x_max),
                        "y_max": int(y_max),
                    },
                })

        return bounding_boxes

    @staticmethod
    def _sign(s: float) -> int:
        return -1 if s < 0 else 1

    def _get_environment_number(self, environment_name: str) -> str:
        """Returns a numeric ID for the environment, creating one if needed."""
        try:
            with open("environment_mapping.txt", "r") as file:
                mapping = file.readlines()
            mapping_dict = {
                line.split(",")[0].strip(): line.split(",")[1].strip()
                for line in mapping
            }
        except FileNotFoundError:
            mapping_dict = {}

        if environment_name not in mapping_dict:
            new_num = f"{len(mapping_dict):02d}"
            with open("environment_mapping.txt", "a") as file:
                file.write(f"{environment_name},{new_num}\n")
            mapping_dict[environment_name] = new_num

        return mapping_dict[environment_name]

    def _create_directories(self, environment_name: str, run_number: int) -> tuple[str, str]:
        """Creates output directory structure for a run."""
        environment_number = self._get_environment_number(environment_name)
        base_dir = f"{environment_name}_{environment_number}"
        run_dir = os.path.join(base_dir, f"Run_{run_number}")

        for subdir in ["rgb", "rgb_bb", "seg", "depth", "label"]:
            os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)

        return run_dir, environment_number


def main():
    parser = argparse.ArgumentParser(
        description="Orbit Navigator: Fly a drone in circular orbits to capture "
        "multi-modal data for TWSI dataset generation."
    )
    parser.add_argument(
        "--radius", type=float, default=2, help="Radius of the orbit in meters"
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=0.3,
        help="Altitude of orbit in positive meters above ground",
    )
    parser.add_argument(
        "--speed", type=float, default=3, help="Speed of orbit in meters/second"
    )
    parser.add_argument(
        "--center",
        type=str,
        default="0,0",
        help="x,y direction vector pointing to orbit center (default: 0,0)",
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of 360-degree orbits"
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=30,
        help="Number of snapshots to capture per orbit",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="Test",
        help="Name of the UE4 environment",
    )
    parser.add_argument(
        "--run", type=int, default=1, help="Run number for output naming"
    )
    args = parser.parse_args()

    nav = OrbitNavigator(
        radius=args.radius,
        altitude=args.altitude,
        speed=args.speed,
        iterations=args.iterations,
        center=args.center.split(","),
        snapshots=args.snapshots,
        environment_name=args.environment,
        run_number=args.run,
    )
    nav.start()


if __name__ == "__main__":
    main()
