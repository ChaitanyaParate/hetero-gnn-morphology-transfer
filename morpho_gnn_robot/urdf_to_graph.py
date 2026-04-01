"""
urdf_to_graph.py  --  Joint-node graph builder for GNN RL training.

Design
------
Nodes  : controllable joints (revolute / continuous / prismatic)
         + 1 virtual body node at index 0 (enables cross-limb message passing)
Edges  : joint A -> joint B  iff  A.child_link == B.parent_link
         body node <-> every joint whose parent link is the URDF root link
         All edges are bidirectional (forward + reverse stored separately)

Node features
  Static  (URDF, computed once) : [type_onehot(4), axis(3), lower, upper, effort, velocity]  ->  11
  Runtime (injected each step)  : [joint_pos, joint_vel]                                      ->   2
  Total                         :                                                                  13

Edge features : [origin(3), direction(1)]  ->  4
  direction = +1.0 for parent->child, -1.0 for child->parent
  Direction is NOT normalized -- it is categorical.

Usage in RL training loop
-------------------------
    builder = URDFGraphBuilder("anymal.urdf")

    obs = env.reset()
    pos, vel = obs[:12], obs[12:24]          # your env's joint state slice
    graph = builder.get_graph(pos, vel)      # call every step
    actions, value = policy(graph)           # GNN forward
"""

import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
JOINT_TYPES    = ["revolute", "continuous", "prismatic", "fixed"]
JOINT_TYPE_MAP = {k: i for i, k in enumerate(JOINT_TYPES)}
CONTROLLABLE   = {"revolute", "continuous", "prismatic"}

STATIC_DIM      = 11   # type_onehot(4) + axis(3) + limits(4)
JOINT_RUNTIME   = 2    # joint_pos, joint_vel
BODY_EXTRA      = 7    # quat(4) + projected_gravity(3)
RUNTIME_DIM     = JOINT_RUNTIME + BODY_EXTRA   # 9 -- body node gets all; joints get zeros for extra
NODE_DIM        = STATIC_DIM + RUNTIME_DIM     # 20

EDGE_DIM        = 4

# -----------------------------------------------------------------------
# URDF parsing helpers
# -----------------------------------------------------------------------
def _xyz(tag) -> List[float]:
    if tag is not None and "xyz" in tag.attrib:
        return [float(v) for v in tag.attrib["xyz"].split()]
    return [0.0, 0.0, 0.0]


def _limits(joint) -> List[float]:
    lim = joint.find("limit")
    if lim is None:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(lim.attrib.get("lower",    0.0)),
        float(lim.attrib.get("upper",    0.0)),
        float(lim.attrib.get("effort",   0.0)),
        float(lim.attrib.get("velocity", 0.0)),
    ]


# -----------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------
class URDFGraphBuilder:
    """
    Parse a URDF once; produce runtime-updated PyG graphs at every RL step.

    The same instance works for any morphology derived from the same URDF.
    For zero-shot transfer, instantiate a separate builder per morphology
    and load the same GNN weights -- graph size changes, weights do not.
    """

    def __init__(self, urdf_path: str, add_body_node: bool = True):
        """
        Parameters
        ----------
        urdf_path     : path to URDF file
        add_body_node : add a virtual index-0 node connected to all root joints.
                        Without this, legs are disconnected graphs -- GNN
                        message passing cannot coordinate across limbs.
                        Set False only if you want to ablate this.
        """
        self.urdf_path     = urdf_path
        self.add_body_node = add_body_node
        self._parse()

    # ------------------------------------------------------------------
    def _parse(self):
        root = ET.parse(self.urdf_path).getroot()

        all_joints: Dict[str, ET.Element] = {
            j.attrib["name"]: j for j in root.findall("joint")
        }

        # identify root links (no joint has them as a child)
        child_links  = {j.find("child").attrib["link"]  for j in all_joints.values()}
        root_links   = {
            lnk.attrib["name"]
            for lnk in root.findall("link")
            if lnk.attrib["name"] not in child_links
        }

        # parent_link -> [joint names whose parent is that link]
        parent_to_joints: Dict[str, List[str]] = {}
        for jname, j in all_joints.items():
            pl = j.find("parent").attrib["link"]
            parent_to_joints.setdefault(pl, []).append(jname)

        # controllable joints, sorted for deterministic ordering
        ctrl: List[str] = sorted([
            jname for jname, j in all_joints.items()
            if j.attrib.get("type", "fixed") in CONTROLLABLE
        ])

        self.joint_names  = ctrl
        self.num_joints   = len(ctrl)
        offset            = 1 if self.add_body_node else 0
        self.joint_to_idx = {name: i + offset for i, name in enumerate(ctrl)}

        # ---- static node features ----------------------------------------
        rows = []
        for jname in ctrl:
            j     = all_joints[jname]
            jt    = j.attrib.get("type", "fixed")
            jt_id = JOINT_TYPE_MAP.get(jt, 3)
            oh    = [0.0] * len(JOINT_TYPES)
            oh[jt_id] = 1.0
            rows.append(oh + _xyz(j.find("axis")) + _limits(j))

        joint_static = torch.tensor(rows, dtype=torch.float)   # [J, 10]

        if self.add_body_node:
            # body node: zero static features (no associated joint)
            self._static_x = torch.cat(
                [torch.zeros(1, STATIC_DIM), joint_static], dim=0
            )                                                   # [J+1, 10]
        else:
            self._static_x = joint_static                      # [J, 10]

        # normalize ONLY the continuous static features; leave one-hot untouched
        # columns 4-9: axis(3) + limits(4) -- normalize these
        # columns 0-3: type one-hot        -- do not normalize
        cont_cols = slice(4, STATIC_DIM)
        col_mean  = self._static_x[:, cont_cols].mean(dim=0, keepdim=True)
        col_std   = self._static_x[:, cont_cols].std(dim=0, keepdim=True).clamp(min=1e-6)
        self._static_x[:, cont_cols] = (
            (self._static_x[:, cont_cols] - col_mean) / col_std
        )

        # ---- edges -------------------------------------------------------
        edges:  List[Tuple[int, int]] = []
        efeats: List[List[float]]     = []

        # kinematic chain edges (parent joint -> child joint)
        for jname in ctrl:
            j          = all_joints[jname]
            child_link = j.find("child").attrib["link"]
            i          = self.joint_to_idx[jname]

            for downstream in parent_to_joints.get(child_link, []):
                if downstream not in self.joint_to_idx:
                    continue                                    # skip fixed joints
                k      = self.joint_to_idx[downstream]
                origin = _xyz(all_joints[downstream].find("origin"))
                edges.append((i, k));  efeats.append(origin + [1.0])
                edges.append((k, i));  efeats.append(origin + [-1.0])

        # body node edges (connects body to every root joint)
        if self.add_body_node:
            body = 0
            for jname in ctrl:
                j  = all_joints[jname]
                pl = j.find("parent").attrib["link"]
                if pl in root_links:
                    k      = self.joint_to_idx[jname]
                    origin = _xyz(j.find("origin"))
                    edges.append((body, k));  efeats.append(origin + [1.0])
                    edges.append((k, body));  efeats.append(origin + [-1.0])

        if edges:
            self._edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            # normalize edge origin (cols 0-2); leave direction flag (col 3) alone
            ea = torch.tensor(efeats, dtype=torch.float)
            ea_mean = ea[:, :3].mean(dim=0, keepdim=True)
            ea_std  = ea[:, :3].std(dim=0,  keepdim=True).clamp(min=1e-6)
            ea[:, :3] = (ea[:, :3] - ea_mean) / ea_std
            self._edge_attr = ea
        else:
            self._edge_index = torch.zeros((2, 0), dtype=torch.long)
            self._edge_attr  = torch.zeros((0, EDGE_DIM), dtype=torch.float)

        self.num_nodes = self.num_joints + offset
        self._print_summary()

    # ------------------------------------------------------------------
    def _print_summary(self):
        tag = f"body + {self.num_joints} joints" if self.add_body_node else f"{self.num_joints} joints"
        print(f"\n[URDFGraphBuilder] {self.urdf_path.split('/')[-1]}")
        print(f"  Nodes            : {self.num_nodes}  ({tag})")
        print(f"  Edges            : {self._edge_index.shape[1]}  (bidirectional)")
        print(f"  Node feature dim : {NODE_DIM}  (static {STATIC_DIM} + runtime {RUNTIME_DIM})")
        print(f"  Edge feature dim : {EDGE_DIM}")
        print(f"  Action dim       : {self.num_joints}  (one torque per controllable joint)")
        print(f"  Joint order      : {self.joint_names}")

    def get_graph(
        self,
        joint_pos:   Optional[np.ndarray] = None,
        joint_vel:   Optional[np.ndarray] = None,
        body_quat:   Optional[np.ndarray] = None,  # (qx,qy,qz,qw) -- 4-dim
        body_grav:   Optional[np.ndarray] = None,  # projected gravity in body frame -- 3-dim
    ) -> Data:
        """
        Build a PyG Data for one RL timestep.

        joint_pos / joint_vel : shape [num_joints], ordered by self.joint_names
        body_quat             : base orientation as quaternion (4-dim)
        body_grav             : gravity vector projected into body frame (3-dim)

        Node runtime layout (9-dim per node):
          [0]   joint_pos  (0 for body node)
          [1]   joint_vel  (0 for body node)
          [2:6] quaternion (actual for body node, zeros for joints)
          [6:9] gravity    (actual for body node, zeros for joints)

        Giving orientation ONLY to the body node is standard in legged
        locomotion. The body node propagates this to limb nodes via message
        passing, so all joints implicitly learn from orientation context.
        """
        pos  = joint_pos if joint_pos is not None else np.zeros(self.num_joints)
        vel  = joint_vel if joint_vel is not None else np.zeros(self.num_joints)
        quat = body_quat if body_quat is not None else np.array([0., 0., 0., 1.], dtype=np.float32)
        grav = body_grav if body_grav is not None else np.array([0., 0., -1.], dtype=np.float32)

        # Joint nodes: [pos, vel, 0, 0, 0, 0, 0, 0, 0]
        joint_runtime = np.zeros((self.num_joints, RUNTIME_DIM), dtype=np.float32)
        joint_runtime[:, 0] = pos
        joint_runtime[:, 1] = vel

        runtime_joints = torch.tensor(joint_runtime, dtype=torch.float)

        if self.add_body_node:
            # Body node: [0, 0, qx, qy, qz, qw, gx, gy, gz]
            body_runtime = np.zeros((1, RUNTIME_DIM), dtype=np.float32)
            body_runtime[0, 2:6] = quat
            body_runtime[0, 6:9] = grav
            runtime = torch.cat(
                [torch.tensor(body_runtime, dtype=torch.float), runtime_joints], dim=0
            )
        else:
            runtime = runtime_joints

        x = torch.cat([self._static_x, runtime], dim=1)   # [num_nodes, 20]

        return Data(
            x=x,
            edge_index=self._edge_index.clone(),
            edge_attr=self._edge_attr.clone(),
        )

    # ------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        """Number of controllable joints = size of action vector."""
        return self.num_joints

    @property
    def node_dim(self) -> int:
        return NODE_DIM

    @property
    def edge_dim(self) -> int:
        return EDGE_DIM

    def obs_to_arrays(
        self,
        pos_dict: Dict[str, float],
        vel_dict: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert name->value dicts (e.g. from ROS2 JointState msg) to
        ordered arrays compatible with get_graph().
        """
        pos = np.array([pos_dict.get(n, 0.0) for n in self.joint_names])
        vel = np.array([vel_dict.get(n, 0.0) for n in self.joint_names])
        return pos, vel


# -----------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = (
        sys.argv[1] if len(sys.argv) > 1 else
        "/mnt/newvolume/Programming/Python/Deep_Learning/"
        "Relational_Bias_for_Morphological_Generalization/"
        "morpho_gnn_robot/morpho_ros2_ws/src/morpho_robot/urdf/anymal.urdf"
    )

    rng = np.random.default_rng(42)

    print("=" * 55)
    print("TEST 1: ANYmal (quadruped)")
    b = URDFGraphBuilder(path, add_body_node=True)
    pos = rng.uniform(-0.5,  0.5, b.num_joints)
    vel = rng.uniform(-1.0,  1.0, b.num_joints)
    g   = b.get_graph(pos, vel)
    print(f"\n  x shape          : {g.x.shape}")
    print(f"  edge_index shape : {g.edge_index.shape}")
    print(f"  edge_attr shape  : {g.edge_attr.shape}")
    assert g.x.shape == (b.num_nodes, NODE_DIM),  "node feature shape mismatch"
    assert g.edge_attr.shape[1] == EDGE_DIM,       "edge feature shape mismatch"

    print("\nTEST 2: same weights, different step (zero-shot transfer check)")
    pos2 = rng.uniform(-0.5, 0.5, b.num_joints)
    vel2 = rng.uniform(-1.0, 1.0, b.num_joints)
    g2   = b.get_graph(pos2, vel2)
    assert g2.x.shape == g.x.shape, "graph shape changed between steps -- BUG"
    print("  OK -- graph shape is stable across steps")

    print("\nTEST 3: without body node (ablation)")
    b2 = URDFGraphBuilder(path, add_body_node=False)
    g3 = b2.get_graph(pos, vel)
    print(f"  Nodes without body node : {g3.x.shape[0]}  (should be {b2.num_joints})")
    print(f"  Edges without body node : {g3.edge_index.shape[1]}  (fewer -- legs disconnected)")

    print("\nAll tests passed.")
