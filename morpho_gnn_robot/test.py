import pybullet as p, pybullet_data
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1/240)
p.loadURDF("plane.urdf")

URDF = "/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/morpho_ros2_ws/src/morpho_robot/urdf/anymal.urdf"
robot = p.loadURDF(URDF, [0, 0, 0.8], useFixedBase=False)

joint_map = {}
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    joint_map[info[1].decode()] = i

for name, idx in joint_map.items():
    if "HAA" in name: p.resetJointState(robot, idx, 0.0)
    elif "HFE" in name: p.resetJointState(robot, idx, 0.6)
    elif "KFE" in name: p.resetJointState(robot, idx, -1.2)

for _ in range(240): p.stepSimulation()

pos, _ = p.getBasePositionAndOrientation(robot)
print(f"Base height after settle: {pos[2]:.4f} m")
print(f"Expected: ~0.50 m for standing ANYmal")
p.disconnect()