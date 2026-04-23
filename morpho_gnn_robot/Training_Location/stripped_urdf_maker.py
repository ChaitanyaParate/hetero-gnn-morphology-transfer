import xml.etree.ElementTree as ET
import shutil, os
SRC = '/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/morpho_ros2_ws/src/morpho_robot/urdf'
URDF_IN = f'{SRC}/anymal.urdf'
URDF_OUT = '/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/Training_Location/anymal_stripped.urdf'

def strip_visual_meshes(urdf_in, urdf_out):
    tree = ET.parse(urdf_in)
    root = tree.getroot()
    removed = 0
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            link.remove(visual)
            removed += 1
    tree.write(urdf_out)
    print(f'Stripped {removed} <visual> elements. Saved to {urdf_out}')
strip_visual_meshes(URDF_IN, URDF_OUT)
URDF = URDF_OUT