from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ami_active_sensing.data import discover_eveyes_scenes, load_scene_imu, load_scene_rgb


class EveEyeDatasetTests(unittest.TestCase):
    def test_discovery_and_lazy_load_preserve_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = root / "train" / "Scene_001" / "cam_static_obj_moving"
            (sequence / "rgb").mkdir(parents=True)
            for index in range(2):
                frame = np.zeros((4, 6, 3), dtype=np.uint8)
                frame[..., index] = 255
                self.assertTrue(cv2.imwrite(str(sequence / "rgb" / f"{index:06d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))
            (sequence / "timestamps.csv").write_text("frame_id,timestamp_s\n0,0.0\n1,0.1\n", encoding="utf-8")
            (sequence / "camera_intrinsics.json").write_text(json.dumps({"width": 6, "height": 4, "fx": 5, "fy": 5, "cx": 2.5, "cy": 1.5}), encoding="utf-8")
            (sequence / "motion_config.json").write_text(json.dumps({"motion_regime": "CAM_STATIC_OBJ_MOVING"}), encoding="utf-8")
            with (sequence / "imu.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp_s", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s", "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"])
                writer.writerow([0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 9.8])
                writer.writerow([0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 9.8])
            scenes = discover_eveyes_scenes(root)
            self.assertEqual(len(scenes), 1)
            scene = scenes[0]
            self.assertEqual(scene.sequence_id, "Scene_001_CAM_STATIC_OBJ_MOVING")
            frames = load_scene_rgb(scene)
            self.assertEqual(frames.shape, (2, 4, 6, 3))
            imu = load_scene_imu(scene)
            self.assertTrue(np.allclose(imu.gyro[:, 2], [0.1, 0.2]))


if __name__ == "__main__":
    unittest.main()
