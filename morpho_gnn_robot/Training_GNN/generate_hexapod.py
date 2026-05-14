import xml.etree.ElementTree as ET
import copy
import os
_DIR = os.path.dirname(os.path.abspath(__file__))
input_urdf = os.path.join(_DIR, 'anymal_stripped.urdf')
output_urdf = os.path.join(_DIR, 'hexapod_anymal.urdf')
tree = ET.parse(input_urdf)
root = tree.getroot()

def clone_and_rename_prefix(parent, original_prefix, new_prefix, origin_x_offset=0.0):
    new_elements = []
    for elem in list(root):
        if 'name' in elem.attrib and elem.attrib['name'].startswith(original_prefix):
            new_elem = copy.deepcopy(elem)
            new_elem.attrib['name'] = new_elem.attrib['name'].replace(original_prefix, new_prefix)
            if new_elem.tag == 'joint':
                for child in new_elem:
                    if child.tag == 'child':
                        child.attrib['link'] = child.attrib['link'].replace(original_prefix, new_prefix)
                    elif child.tag == 'parent' and child.attrib['link'].startswith(original_prefix):
                        child.attrib['link'] = child.attrib['link'].replace(original_prefix, new_prefix)
                    elif child.tag == 'origin' and 'xyz' in child.attrib:
                        if new_elem.attrib['name'].endswith('_HAA'):
                            xyz = child.attrib['xyz'].split()
                            xyz[0] = str(float(xyz[0]) + origin_x_offset)
                            child.attrib['xyz'] = ' '.join(xyz)
            new_elements.append(new_elem)
    for elem in list(root):
        if elem.tag == 'gazebo' and 'reference' in elem.attrib and elem.attrib['reference'].startswith(original_prefix):
            new_elem = copy.deepcopy(elem)
            new_elem.attrib['reference'] = new_elem.attrib['reference'].replace(original_prefix, new_prefix)
            new_elements.append(new_elem)
    for gazebo in root.findall('gazebo'):
        for plugin in gazebo.findall('plugin'):
            for joint_name in plugin.findall('joint_name'):
                if joint_name.text and joint_name.text.startswith(original_prefix):
                    new_plugin = copy.deepcopy(plugin)
                    new_plugin.find('joint_name').text = new_plugin.find('joint_name').text.replace(original_prefix, new_prefix)
                    topic = new_plugin.find('topic')
                    if topic is not None and topic.text:
                        topic.text = topic.text.replace(original_prefix, new_prefix)
                    gazebo.append(new_plugin)
    for ne in new_elements:
        root.append(ne)
clone_and_rename_prefix(root, 'LF_', 'LM_', origin_x_offset=-0.277)
clone_and_rename_prefix(root, 'RF_', 'RM_', origin_x_offset=-0.277)
with open(output_urdf, 'wb') as f:
    f.write(b'<?xml version="1.0" ?>\n')
    tree.write(f, encoding='utf-8')
print(f'Generated {output_urdf} with 18 joints successfully!')