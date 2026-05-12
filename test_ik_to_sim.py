import numpy as np
import pybullet as p
import time
import math
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


# -----------------------------
# Testpunkter utan kamera
# -----------------------------
test_points = [
    # shoulder, elbow, wrist
    ([0, 0, 0], [0, -1, 0], [0, -2, 0]),       # rak arm
    ([0, 0, 0], [0, -1, 0], [0.5, -1.5, 0]),   # lite böjd
    ([0, 0, 0], [0, -1, 0], [1.0, -1.0, 0]),   # mer böjd
]

ik = InverseKinematics()
ik.start()

q_list = []

for shoulder, elbow, wrist in test_points:
    angles = ik.calculate_arm_angles(shoulder, elbow, wrist)

    if angles is not None:
        q = angles["q_rad"]
        q = clamp_dmp_vector(q)
        q_list.append(q)
        print("q deg:", np.degrees(q))

ik.stop()

q_traj = np.array(q_list)

# -----------------------------
# Starta simulator
# -----------------------------
sim_dir = Path("sim")

p.connect(p.GUI)
p.setGravity(0, 0, 0)
p.setAdditionalSearchPath(str(sim_dir))

base_orn = p.getQuaternionFromEuler([0, 0, 0])

robot = p.loadURDF(
    "arm/left_arm.urdf",
    basePosition=[0, 0, 0],
    baseOrientation=base_orn,
    useFixedBase=True,
)

sh_rotz = joint_index(robot, "jLeftShoulder_rotz")
sh_rotx = joint_index(robot, "jLeftShoulder_rotx")
sh_roty = joint_index(robot, "jLeftShoulder_roty")
elbow_roty = joint_index(robot, "jLeftElbow_roty")

# -----------------------------
# Disable self collisions
# -----------------------------
num_joints = p.getNumJoints(robot)

for i in range(-1, num_joints):
    for j in range(-1, num_joints):
        p.setCollisionFilterPair(
            robot,
            robot,
            i,
            j,
            enableCollision=0
        )

# -----------------------------
# Test pose
# -----------------------------
elbow = 0.8
sh_flex = 0.0
sh_abd = 0.0
sh_rot = 0.0

for elbow in np.linspace(0.0, 1.05, 50):
    p.resetJointState(robot, sh_rotz, 0.0)
    p.resetJointState(robot, sh_roty, 0.0)
    p.resetJointState(robot, sh_rotx, 0.0)
    p.resetJointState(robot, elbow_roty, elbow) #Armbåge

    time.sleep(0.05)

for sh_flex in np.linspace(0.0, 1.39, 50):
    p.resetJointState(robot, sh_rotz, 0.0)
    p.resetJointState(robot, sh_roty, sh_flex) #Axel Fram
    p.resetJointState(robot, sh_rotx, 0.0)
    p.resetJointState(robot, elbow_roty, elbow)

    time.sleep(0.05)

for sh_abd in np.linspace(0.0, -0.69, 50):
    p.resetJointState(robot, sh_rotz, sh_abd)
    p.resetJointState(robot, sh_roty, sh_flex)
    p.resetJointState(robot, sh_rotx, sh_rot) #roation
    p.resetJointState(robot, elbow_roty, elbow)

    time.sleep(0.05)

for sh_rot in np.linspace(0.0, -0.69, 50):
    p.resetJointState(robot, sh_rotz, sh_abd) #Axel sida inverterad?
    p.resetJointState(robot, sh_roty, sh_flex)
    p.resetJointState(robot, sh_rotx, sh_rot)
    p.resetJointState(robot, elbow_roty, elbow)

    time.sleep(0.05)

for sh_rot in np.linspace(-0.69, 0.69, 50):
    p.resetJointState(robot, sh_rotz, sh_abd)
    p.resetJointState(robot, sh_roty, sh_flex)
    p.resetJointState(robot, sh_rotx, sh_rot)
    p.resetJointState(robot, elbow_roty, elbow)

    time.sleep(0.05)



# -----------------------------
# Keep window alive
# -----------------------------
while p.isConnected():
    time.sleep(0.01)

print("Done. Close window to exit.")
while p.isConnected():
    p.stepSimulation()
    time.sleep(0.01)