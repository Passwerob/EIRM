#!/usr/bin/env python3
"""Extract a DSEC ROS1 IMU topic to the simulator CSV convention."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract sensor_msgs/Imu from a DSEC ROS1 bag.")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", default="/imu/data")
    parser.add_argument("--start", type=float, required=True, help="Inclusive start time in absolute seconds.")
    parser.add_argument("--end", type=float, required=True, help="Inclusive end time in absolute seconds.")
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("end must be greater than start")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS1_NOETIC)
    count = 0
    with Reader(args.bag) as reader, args.output.open("w", newline="", encoding="utf-8") as handle:
        connections = [connection for connection in reader.connections if connection.topic == args.topic and connection.msgtype == "sensor_msgs/msg/Imu"]
        if not connections:
            available = ", ".join(f"{connection.topic} ({connection.msgtype})" for connection in reader.connections)
            raise ValueError(f"No sensor_msgs/Imu topic {args.topic!r}; available: {available}")
        writer = csv.writer(handle)
        writer.writerow(["timestamp_us", "ax", "ay", "az", "gx", "gy", "gz"])
        for connection, _, raw in reader.messages(connections=connections, start=int(args.start * 1e9), stop=int(args.end * 1e9)):
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            timestamp_us = (message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec) / 1000.0
            timestamp = timestamp_us * 1e-6
            if not args.start <= timestamp <= args.end:
                continue
            writer.writerow([f"{timestamp_us:.3f}", f"{message.linear_acceleration.x:.9g}", f"{message.linear_acceleration.y:.9g}", f"{message.linear_acceleration.z:.9g}", f"{message.angular_velocity.x:.9g}", f"{message.angular_velocity.y:.9g}", f"{message.angular_velocity.z:.9g}"])
            count += 1
    print(f"wrote {count} IMU samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
