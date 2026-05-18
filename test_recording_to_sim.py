import json
import time
import numpy as np
import pybullet as p
from pathlib import Path

from inverse_kinematics import InverseKinematics
from sim.joint_limits import clamp_dmp_vector


def joint_index(body_uid, joint_name):
    for i in range(p.getNumJoints(body_uid)):
        info = p.getJointInfo(body_uid, i)
        name = info[1].decode("utf-8")
        if name == joint_name:
            return i
    raise KeyError(f"Joint not found: {joint_name}")


def set_pose(robot, sh_rotz, sh_roty, sh_rotx, elbow_roty,
             sh_rot=0.0, sh_flex=0.0, sh_abd=0.0, elbow=0.0):

    p.resetJointState(robot, sh_rotz, -sh_abd)   # abduktion
    p.resetJointState(robot, sh_roty, sh_flex)   # flexion
    p.resetJointState(robot, sh_rotx, sh_rot)    # rotation
    p.resetJointState(robot, elbow_roty, elbow)  # elbow


# -----------------------------
# Load recording
# -----------------------------
with open("Data/Training/ID1run4.json", "r") as f:
    recording = json.load(f)

frames = recording["data"]

print("Loaded frames:", len(frames))

# -----------------------------
# Start simulator
# -----------------------------
p.connect(p.GUI)
p.setGravity(0, 0, 0)

robot = p.loadURDF(
    "sim/arm/left_arm.urdf",
    basePosition=[0, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
)

sh_rotz = joint_index(robot, "jLeftShoulder_rotz")
sh_rotx = joint_index(robot, "jLeftShoulder_rotx")
sh_roty = joint_index(robot, "jLeftShoulder_roty")
elbow_roty = joint_index(robot, "jLeftElbow_roty")

# Stäng av self-collision
num_joints = p.getNumJoints(robot)
for i in range(-1, num_joints):
    for j in range(-1, num_joints):
        p.setCollisionFilterPair(robot, robot, i, j, enableCollision=0)

# -----------------------------
# IK setup
# -----------------------------
ik = InverseKinematics()
ik.start()

# -----------------------------
# Playback
# -----------------------------
for idx, frame in enumerate(frames):
    shoulder = frame["shoulder"]
    elbow = frame["elbow"]

    # Wrist från handens punkt 0
    if "hand" not in frame or len(frame["hand"]) == 0:
        continue

    wrist = [
        frame["hand"][0]["x"],
        frame["hand"][0]["y"],
        frame["hand"][0]["depth_m"]
    ]

    # Om någon punkt saknar z, hoppa över frame
    if shoulder[2] is None or elbow[2] is None or wrist[2] is None:
        continue

    angles = ik.calculate_arm_angles(shoulder, elbow, wrist)

    if angles is None:
        continue

    q = angles["q_rad"]
    q = clamp_dmp_vector(q)
    print(f"Frame {idx}: q_deg = {np.degrees(q)}")

    set_pose(
        robot,
        sh_rotz,
        sh_roty,
        sh_rotx,
        elbow_roty,
        elbow=float(q[0]),
        sh_flex=float(q[1]),
        sh_abd=float(q[2]),
        sh_rot=float(q[3])
    )

    if idx % 10 == 0:
        print("Frame:", idx, "q_deg:", np.degrees(q))

    time.sleep(0.05)

ik.stop()

print("Playback done. Close window to exit.")

while p.isConnected():
    time.sleep(0.01)