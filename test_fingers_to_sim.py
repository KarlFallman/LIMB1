import time
import numpy as np
import pybullet as p


def set_finger(joints, angle):
    for j in joints:
        p.resetJointState(robot, j, angle)


p.connect(p.GUI)
p.setGravity(0, 0, 0)

robot = p.loadURDF(
    "sim/arm/left_arm.urdf",
    basePosition=[0, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
)

thumb_joints  = [9, 10, 11]
index_joints  = [12, 13, 14]
middle_joints = [15, 16, 17]
ring_joints   = [19, 20, 21]
pinky_joints  = [23, 24, 25]

pinky_anchor = pinky_joints[0]
pinky_bend_joints = pinky_joints[1:]

ring_anchor = ring_joints[0]
ring_bend_joints = ring_joints[1:]

all_finger_joints = (
    thumb_joints
    + index_joints
    + middle_joints
    + ring_joints
    + pinky_joints
)


def set_hand_grip(grip):
    grip = np.clip(grip, 0.0, 1.0)

    index_angle = -grip * 0.8
    middle_angle = -grip * 0.8
    ring_angle = -grip * 0.8

    pinky_anchor_angle = -grip * 0.8
    pinky_angle = -grip * 0.8

    p.resetJointState(robot, pinky_anchor, pinky_anchor_angle)

    ring_anchor_angle = -grip * 0.8
    ring_angle = -grip * 0.8

    p.resetJointState(robot, ring_anchor, ring_anchor_angle)

    thumb_angles = [
        grip * 0.5,
        grip * 0.6,
        grip * 0.8,
    ]
   
    for joint, angle in zip(thumb_joints, thumb_angles):
        p.resetJointState(robot, joint, angle)

    for j in index_joints:
        p.resetJointState(robot, j, index_angle)

    for j in middle_joints:
        p.resetJointState(robot, j, middle_angle)

    for j in ring_bend_joints:
        p.resetJointState(robot, j, ring_angle)

    for j in pinky_bend_joints:
        p.resetJointState(robot, j, pinky_angle)
    
   

# Testa öppna/stäng
while True:

    for grip in np.linspace(0.0, 1.0, 50):
        set_hand_grip(grip)
        time.sleep(0.03)

    for grip in np.linspace(1.0, 0.0, 50):
        set_hand_grip(grip)
        time.sleep(0.03)