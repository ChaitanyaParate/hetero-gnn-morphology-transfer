import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple
JOINT_TYPES = ['revolute', 'continuous', 'prismatic', 'fixed']
JOINT_TYPE_MAP = {k: i for i, k in enumerate(JOINT_TYPES)}
CONTROLLABLE = {'revolute', 'continuous', 'prismatic'}
STATIC_DIM = 11
JOINT_RUNTIME = 2
BODY_EXTRA = 15  # Includes 2 extra for command (vx, wy)
RUNTIME_DIM = JOINT_RUNTIME + BODY_EXTRA
NODE_DIM = STATIC_DIM + RUNTIME_DIM
EDGE_DIM = 4
NODE_ROLE_BODY = 0
NODE_ROLE_HAA = 1
NODE_ROLE_HFE = 2
NODE_ROLE_KFE = 3
NODE_ROLE_GENERIC = 4
NUM_NODE_ROLES = 5
# Multi-vendor keyword-to-role mapping.
# Priority: first match wins (ordered from most specific to most generic).
# Supports: ANYmal (HAA/HFE/KFE), Unitree (hip/thigh/calf), and generic aliases.
KEYWORD_ROLE_MAP: Dict[str, int] = {
    # ANYmal-style (exact suffix match, e.g. LF_HAA, RF_KFE)
    'HAA': NODE_ROLE_HAA,
    'HFE': NODE_ROLE_HFE,
    'KFE': NODE_ROLE_KFE,
    # Unitree-style (e.g. FR_hip_joint, FR_thigh_joint, FR_calf_joint)
    'HIP': NODE_ROLE_HAA,
    'THIGH': NODE_ROLE_HFE,
    'CALF': NODE_ROLE_KFE,
    # Generic aliases for other vendors
    'ABDUCT': NODE_ROLE_HAA,
    'FLEX': NODE_ROLE_HFE,
    'KNEE': NODE_ROLE_KFE,
    'SHOULDER': NODE_ROLE_HAA,
    'ELBOW': NODE_ROLE_KFE,
}

def _joint_role(joint_name: str) -> int:
    """Map a joint name to a node role using multi-vendor keyword search.

    Checks the full joint name (upper-cased) for each keyword in
    KEYWORD_ROLE_MAP. The first matching keyword wins. Falls back to
    NODE_ROLE_GENERIC for unrecognised joints.
    """
    name_upper = joint_name.upper()
    for keyword, role in KEYWORD_ROLE_MAP.items():
        if keyword in name_upper:
            return role
    return NODE_ROLE_GENERIC

def _xyz(tag) -> List[float]:
    if tag is not None and 'xyz' in tag.attrib:
        return [float(v) for v in tag.attrib['xyz'].split()]
    return [0.0, 0.0, 0.0]

def _limits(joint) -> List[float]:
    lim = joint.find('limit')
    if lim is None:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(lim.attrib.get('lower', 0.0)), float(lim.attrib.get('upper', 0.0)), float(lim.attrib.get('effort', 0.0)), float(lim.attrib.get('velocity', 0.0))]

class URDFGraphBuilder:

    def __init__(self, urdf_path: str, add_body_node: bool=True):
        self.urdf_path = urdf_path
        self.add_body_node = add_body_node
        self._parse()

    def _parse(self):
        root = ET.parse(self.urdf_path).getroot()
        all_joints: Dict[str, ET.Element] = {j.attrib['name']: j for j in root.findall('joint')}
        child_links = {j.find('child').attrib['link'] for j in all_joints.values()}
        root_links = {lnk.attrib['name'] for lnk in root.findall('link') if lnk.attrib['name'] not in child_links}
        parent_to_joints: Dict[str, List[str]] = {}
        for jname, j in all_joints.items():
            pl = j.find('parent').attrib['link']
            parent_to_joints.setdefault(pl, []).append(jname)
        ctrl: List[str] = sorted([jname for jname, j in all_joints.items() if j.attrib.get('type', 'fixed') in CONTROLLABLE])
        self.joint_names = ctrl
        self.num_joints = len(ctrl)
        offset = 1 if self.add_body_node else 0
        self.joint_to_idx = {name: i + offset for i, name in enumerate(ctrl)}
        joint_roles = [_joint_role(jname) for jname in ctrl]
        if self.add_body_node:
            all_roles = [NODE_ROLE_BODY] + joint_roles
        else:
            all_roles = joint_roles
        self._node_types = torch.tensor(all_roles, dtype=torch.long)
        rows = []
        for jname in ctrl:
            j = all_joints[jname]
            jt = j.attrib.get('type', 'fixed')
            jt_id = JOINT_TYPE_MAP.get(jt, 3)
            oh = [0.0] * len(JOINT_TYPES)
            oh[jt_id] = 1.0
            rows.append(oh + _xyz(j.find('axis')) + _limits(j))
        joint_static = torch.tensor(rows, dtype=torch.float)
        if self.add_body_node:
            self._static_x = torch.cat([torch.zeros(1, STATIC_DIM), joint_static], dim=0)
        else:
            self._static_x = joint_static
        cont_cols = slice(4, STATIC_DIM)
        col_mean = self._static_x[:, cont_cols].mean(dim=0, keepdim=True)
        col_std = self._static_x[:, cont_cols].std(dim=0, keepdim=True).clamp(min=1e-06)
        self._static_x[:, cont_cols] = (self._static_x[:, cont_cols] - col_mean) / col_std
        edges: List[Tuple[int, int]] = []
        efeats: List[List[float]] = []
        for jname in ctrl:
            j = all_joints[jname]
            child_link = j.find('child').attrib['link']
            i = self.joint_to_idx[jname]
            for downstream in parent_to_joints.get(child_link, []):
                if downstream not in self.joint_to_idx:
                    continue
                k = self.joint_to_idx[downstream]
                origin = _xyz(all_joints[downstream].find('origin'))
                edges.append((i, k))
                efeats.append(origin + [1.0])
                edges.append((k, i))
                efeats.append(origin + [-1.0])
        if self.add_body_node:
            body = 0
            for jname in ctrl:
                j = all_joints[jname]
                pl = j.find('parent').attrib['link']
                if pl in root_links:
                    k = self.joint_to_idx[jname]
                    origin = _xyz(j.find('origin'))
                    edges.append((body, k))
                    efeats.append(origin + [1.0])
                    edges.append((k, body))
                    efeats.append(origin + [-1.0])
        if edges:
            self._edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            ea = torch.tensor(efeats, dtype=torch.float)
            ea_mean = ea[:, :3].mean(dim=0, keepdim=True)
            ea_std = ea[:, :3].std(dim=0, keepdim=True).clamp(min=1e-06)
            ea[:, :3] = (ea[:, :3] - ea_mean) / ea_std
            self._edge_attr = ea
        else:
            self._edge_index = torch.zeros((2, 0), dtype=torch.long)
            self._edge_attr = torch.zeros((0, EDGE_DIM), dtype=torch.float)
        self.num_nodes = self.num_joints + offset
        self._print_summary()

    def _print_summary(self):
        tag = f'body + {self.num_joints} joints' if self.add_body_node else f'{self.num_joints} joints'
        role_counts = {}
        for r in self._node_types.tolist():
            role_counts[r] = role_counts.get(r, 0) + 1
        role_names = {NODE_ROLE_BODY: 'body', NODE_ROLE_HAA: 'HAA', NODE_ROLE_HFE: 'HFE', NODE_ROLE_KFE: 'KFE', NODE_ROLE_GENERIC: 'generic'}
        role_str = ', '.join((f'{role_names.get(k, '?')}x{v}' for k, v in sorted(role_counts.items())))
        print(f'\n[URDFGraphBuilder] {self.urdf_path.split('/')[-1]}')
        print(f'  Nodes            : {self.num_nodes}  ({tag})')
        print(f'  Node roles       : {role_str}')
        print(f'  Edges            : {self._edge_index.shape[1]}  (bidirectional)')
        print(f'  Node feature dim : {NODE_DIM}  (static {STATIC_DIM} + runtime {RUNTIME_DIM})')
        print(f'  Edge feature dim : {EDGE_DIM}')
        print(f'  Action dim       : {self.num_joints}  (one torque per controllable joint)')
        print(f'  Joint order      : {self.joint_names}')

    def get_graph(self, joint_pos: Optional[np.ndarray]=None, joint_vel: Optional[np.ndarray]=None, body_quat: Optional[np.ndarray]=None, body_grav: Optional[np.ndarray]=None, body_lin_vel: Optional[np.ndarray]=None, body_ang_vel: Optional[np.ndarray]=None, command: Optional[np.ndarray]=None) -> Data:
        pos = joint_pos if joint_pos is not None else np.zeros(self.num_joints)
        vel = joint_vel if joint_vel is not None else np.zeros(self.num_joints)
        quat = body_quat if body_quat is not None else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        grav = body_grav if body_grav is not None else np.array([0.0, 0.0, -1.0], dtype=np.float32)
        lin_vel = body_lin_vel if body_lin_vel is not None else np.zeros(3, dtype=np.float32)
        ang_vel = body_ang_vel if body_ang_vel is not None else np.zeros(3, dtype=np.float32)
        cmd = command if command is not None else np.zeros(2, dtype=np.float32)
        joint_runtime = np.zeros((self.num_joints, RUNTIME_DIM), dtype=np.float32)
        joint_runtime[:, 0] = pos
        joint_runtime[:, 1] = vel
        joint_runtime[:, 15:17] = cmd  # Broadcast command to all joints
        runtime_joints = torch.tensor(joint_runtime, dtype=torch.float)
        if self.add_body_node:
            body_runtime = np.zeros((1, RUNTIME_DIM), dtype=np.float32)
            body_runtime[0, 2:6] = quat
            body_runtime[0, 6:9] = grav
            body_runtime[0, 9:12] = lin_vel
            body_runtime[0, 12:15] = ang_vel
            body_runtime[0, 15:17] = cmd
            runtime = torch.cat([torch.tensor(body_runtime, dtype=torch.float), runtime_joints], dim=0)
        else:
            runtime = runtime_joints
        x = torch.cat([self._static_x, runtime], dim=1)
        return Data(x=x, edge_index=self._edge_index.clone(), edge_attr=self._edge_attr.clone(), node_types=self._node_types.clone())

    @property
    def action_dim(self) -> int:
        return self.num_joints

    @property
    def node_dim(self) -> int:
        return NODE_DIM

    @property
    def edge_dim(self) -> int:
        return EDGE_DIM

    @property
    def node_roles(self) -> torch.Tensor:
        return self._node_types

    def obs_to_arrays(self, pos_dict: Dict[str, float], vel_dict: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.array([pos_dict.get(n, 0.0) for n in self.joint_names])
        vel = np.array([vel_dict.get(n, 0.0) for n in self.joint_names])
        return (pos, vel)
if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/morpho_ros2_ws/src/morpho_robot/urdf/anymal.urdf'
    rng = np.random.default_rng(42)
    b = URDFGraphBuilder(path, add_body_node=True)
    print('\n  node_roles tensor:', b.node_roles.tolist())
    print('  expected: [0, 1,2,3, 1,2,3, 1,2,3, 1,2,3]  (body, then 4x HAA/HFE/KFE)')
    pos = rng.uniform(-0.5, 0.5, b.num_joints)
    vel = rng.uniform(-1.0, 1.0, b.num_joints)
    g = b.get_graph(pos, vel)
    assert g.x.shape == (b.num_nodes, NODE_DIM), 'node feature shape mismatch'
    assert g.edge_attr.shape[1] == EDGE_DIM, 'edge feature shape mismatch'
    assert g.node_types.shape == (b.num_nodes,), 'node_types shape mismatch'
    assert g.node_types.dtype == torch.long, 'node_types must be LongTensor'
    print('\n  graph.node_types:', g.node_types.tolist())
    print('\nAll tests passed.')