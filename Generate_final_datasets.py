import os
import sys
import shutil
import random
import numpy as np
import xml.etree.cElementTree as ET
import bpy, bpy_extras
if '' not in sys.path:
    sys.path.append('')
from blender_utils import *

scenario_dir = 'Datasets/Scenarios'
model_dir = 'Datasets/3D_models'
background = 'backgrounds/background%d.blend'
datasets_dir = 'Datasets/Depths'
if not os.path.exists(datasets_dir):
    os.mkdir(datasets_dir)
if os.path.exists('tmp'):
    shutil.rmtree('tmp', True)
os.mkdir('tmp')
start_idx = 7000
scenario_names = os.listdir(scenario_dir)


def add_noise_s(Img, Mask):

    row, col = Img.shape
    kernel_size = int(row / 100)
    edge_mask_size = int(row / 4)
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    Mask_exp = cv2.dilate(Mask,kernel,iterations = 1)
    edge = Mask_exp - Mask
    Masks_Mask = np.zeros((row, col))

    coors = np.random.randint(0, row - edge_mask_size, (3, 2))
    for i in range(3):  
        Masks_Mask[coors[i, 0]:coors[i, 0]+edge_mask_size, coors[i, 1]:coors[i, 1]+edge_mask_size] = 1
    edge_mask = (edge * Masks_Mask) > 0
    '''
    f = open('ss.txt', 'a')
    f.write('edge' + str(edge.sum()) + '\n')
    f.write('Masks_Mask' + str(Masks_Mask.sum()) + '\n')
    '''
    mean = 0
    sigma = 0.0165
    for i in range(3):
        gauss = np.random.normal(mean,sigma,(row * 2 / 5,col * 2 / 5))
        gauss = gauss.reshape(row * 2 / 5, col * 2 / 5)
        gauss = cv2.GaussianBlur(gauss, (kernel_size, kernel_size), 0)
        gauss = cv2.resize(gauss, (col, row), interpolation=cv2.INTER_CUBIC)

        gauss_edge = np.random.normal(0.015 ,sigma,(row * 2 / 5,col * 2 / 5))
        gauss_edge = gauss_edge.reshape(row * 2 / 5, col * 2 / 5)
        gauss_edge = cv2.GaussianBlur(gauss_edge, (kernel_size, kernel_size), 0)
        gauss_edge = cv2.resize(gauss_edge, (col, row), interpolation=cv2.INTER_CUBIC)

        gauss_edge = gauss_edge * edge_mask
        gauss_edge = cv2.GaussianBlur(gauss_edge, (kernel_size, kernel_size), 0)
        gauss_edge = cv2.GaussianBlur(gauss_edge, (kernel_size, kernel_size), 0)
        gauss_edge = cv2.GaussianBlur(gauss_edge, (kernel_size, kernel_size), 0)
        # f.write('Masks_Mask' + str(Masks_Mask.sum()) + '\n')
        Img = Img + gauss + gauss_edge
        Img = np.clip(Img, 0, 255)
        Img = cv2.GaussianBlur(Img, (kernel_size, kernel_size), 0)

    return Img

def add_noise(Img, Mask):

    row, col = Img.shape
    kernel_size = int(row / 100)
    edge_mask_size = int(row / 4)
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    Mask_exp = cv2.dilate(Mask,kernel,iterations = 1)
    edge = Mask_exp - Mask
    Masks_Mask = np.zeros((row, col))

    coors = np.random.randint(0, row - edge_mask_size, (3, 2))
    for i in range(3):  
        Masks_Mask[coors[i, 0]:coors[i, 0]+edge_mask_size, coors[i, 1]:coors[i, 1]+edge_mask_size] = 1
    edge_mask = (edge * Masks_Mask) > 0
    '''
    f = open('ss.txt', 'a')
    f.write('edge' + str(edge.sum()) + '\n')
    f.write('Masks_Mask' + str(Masks_Mask.sum()) + '\n')
    '''
    mean = 0
    sigma = 0.0165
    for i in range(3):
        gauss = np.random.normal(mean,sigma,(row * 2 / 5,col * 2 / 5))
        gauss = gauss.reshape(row * 2 / 5, col * 2 / 5)
        gauss = cv2.GaussianBlur(gauss, (kernel_size, kernel_size), 0)
        gauss = cv2.resize(gauss, (col, row), interpolation=cv2.INTER_CUBIC)

        gauss_edge = np.random.normal(0.015 ,sigma,(row * 2 / 5,col * 2 / 5))
        gauss_edge = gauss_edge.reshape(row * 2 / 5, col * 2 / 5)
        gauss_edge = cv2.GaussianBlur(gauss_edge, (kernel_size, kernel_size), 0)
        gauss_edge = cv2.resize(gauss_edge, (col, row), interpolation=cv2.INTER_CUBIC)

        gauss_edge = gauss_edge * edge_mask
        gauss_edge = cv2.GaussianBlur(gauss_edge, (kernel_size, kernel_size), 0)
        gauss_edge = cv2.GaussianBlur(gauss_edge, (kernel_size, kernel_size), 0)
        gauss_edge = cv2.GaussianBlur(gauss_edge, (kernel_size, kernel_size), 0)
        # f.write('Masks_Mask' + str(Masks_Mask.sum()) + '\n')
        Img = Img + gauss + gauss_edge
        Img = np.clip(Img, 0, 255)
        Img = cv2.GaussianBlur(Img, (kernel_size, kernel_size), 0)

    kernel = np.ones((kernel_size,kernel_size),np.uint8)

    '''
    f = open('ss.txt', 'a')
    f.write('edge' + str(edge.sum()) + '\n')
    f.write('Masks_Mask' + str(Masks_Mask.sum()) + '\n')
    '''
    mean = 0
    sigma = 0.0165
    Img = cv2.blur(Img, (7, 7))
    Img = cv2.blur(Img, (7, 7))
    Img = cv2.blur(Img, (5, 5))
    for i in range(3):
        gauss = np.random.normal(mean,sigma,(row * 2 / 5,col * 2 / 5))
        gauss = gauss.reshape(row * 2 / 5, col * 2 / 5)
        gauss = cv2.GaussianBlur(gauss, (kernel_size, kernel_size), 0)
        gauss = cv2.resize(gauss, (col, row), interpolation=cv2.INTER_CUBIC)
        Img = Img + gauss
        Img = cv2.GaussianBlur(Img, (kernel_size, kernel_size), 0)

    return Img
    
def write_xml_single(Filename, objects, bboxes):
    bboxes = bboxes.astype(np.int)
    root = ET.Element('annotations')
    root.text = '\n\t'
    Tmp = ET.SubElement(root, 'folder')
    Tmp.text = 'VOC2007'
    Tmp.tail = '\n\t'
    Tmp = ET.SubElement(root, 'filename')
    Tmp.text = (Filename.split('/')[-1]).split('.')[0]
    Tmp.tail = '\n\t'
    source = ET.SubElement(root, 'source')
    source.text = '\n\t\t'
    source.tail = '\n\t'
    Tmp = ET.SubElement(source, 'database')
    Tmp.text = 'The VOC2007 Database'
    Tmp.tail = '\n\t\t'
    Tmp = ET.SubElement(source, 'annotations')
    Tmp.text = 'PASCAL VOC2007'
    Tmp.tail = '\n\t\t'
    Tmp = ET.SubElement(source, 'image')
    Tmp.text = 'flickr'
    Tmp.tail = '\n\t\t'
    Tmp = ET.SubElement(source, 'flickrid')
    Tmp.text = '325991873'
    Tmp.tail = '\n\t'
    owner = ET.SubElement(root, 'owner')
    owner.text = '\n\t\t'
    owner.tail = '\n\t'
    Tmp = ET.SubElement(owner, 'flickrid')
    Tmp.text = 'archintent louisville'
    Tmp.tail = '\n\t\t'
    Tmp = ET.SubElement(owner, 'name')
    Tmp.text = '?'
    Tmp.tail = '\n\t'
    size = ET.SubElement(root, 'size')
    size.text = '\n\t\t'
    size.tail = '\n\t'
    Tmp = ET.SubElement(size, 'weight')
    Tmp.text = '320'
    Tmp.tail = '\n\t\t'
    Tmp = ET.SubElement(size, 'height')
    Tmp.text = '240'
    Tmp.tail = '\n\t\t'
    Tmp = ET.SubElement(size, 'depth')
    Tmp.text = '3'
    Tmp.tail = '\n\t'
    Tmp = ET.SubElement(root, 'segmented')
    Tmp.text = '0'
    Tmp.tail = '\n\t'
    for i, obj in enumerate(objects):
        Object = ET.SubElement(root, 'object')
        Object.text = '\n\t\t'
        Object.tail = '\n\t'
        Tmp = ET.SubElement(Object, 'name')
        Tmp.text = obj
        Tmp.tail = '\n\t\t'
        Tmp = ET.SubElement(Object, 'pose')
        Tmp.text = 'Unspecified'
        Tmp.tail = '\n\t\t'
        Tmp = ET.SubElement(Object, 'truncated')
        Tmp.text = '0'
        Tmp.tail = '\n\t\t'
        Tmp = ET.SubElement(Object, 'index')
        Tmp.text = str(i + 1)
        Tmp.tail = '\n\t\t'
        Tmp = ET.SubElement(Object, 'difficult')
        Tmp.text = '0'
        Tmp.tail = '\n\t\t'
        bndbox = ET.SubElement(Object, 'bndbox')
        bndbox.text = '\n\t\t\t'
        bndbox.tail = '\n\t'
        Tmp = ET.SubElement(bndbox, 'xmin')
        Tmp.text = str(bboxes[i, 0])
        Tmp.tail = '\n\t\t\t'
        Tmp = ET.SubElement(bndbox, 'ymin')
        Tmp.text = str(bboxes[i, 1])
        Tmp.tail = '\n\t\t\t'
        Tmp = ET.SubElement(bndbox, 'xmax')
        Tmp.text = str(bboxes[i, 2])
        Tmp.tail = '\n\t\t\t'
        Tmp = ET.SubElement(bndbox, 'ymax')
        Tmp.text = str(bboxes[i, 3])
        Tmp.tail = '\n\t\t'
    Object.tail = '\n'
    tree = ET.ElementTree(root)
    tree.write(Filename)

def write_xml_and_depth_and_grasps(Dirname, objects, tasks, bboxes, grasps):

    Dir_list = Dirname.split('/')
    dir_name = Dir_list.pop(-1)
    Dir_list.pop(-1)
    Dir_list.append('tmp')
    Dir_list.append('re%d'%(int(dir_name.split('_')[-1]) - start_idx))
    Dir_list.append('ff0.png')
   
    Mask_path = os.path.join('tmp', 're%d'%(int(dir_name.split('_')[-1]) - start_idx), 'ff0.png')
    Mask_s = cv2.imread(Mask_path)
    Mask = Mask_s[:, :, 0] > 70
    Mask = (Mask * 255).astype(np.uint8)
    
    write_xml_single(os.path.join(Dirname, 'original.xml'), objects, bboxes)
    Dep = cv2.imread(os.path.join(Dirname, 'Image0001.exr'), cv2.IMREAD_UNCHANGED)
    Dep = Dep[:, :, 0]
    bx_l = bboxes[:, 0].min()
    by_l = bboxes[:, 1].min()
    bx_r = bboxes[:, 2].max()
    by_r = bboxes[:, 3].max()
    center_x = (bx_l + bx_r) / 2
    center_y = (by_l + by_r) / 2
    w = bx_r - bx_l
    h = by_r - by_l
    side = max(w, h)

    exp = random.uniform(1.05, 1.3)    
    x_l = int(center_x - side / 2 * exp)
    x_r = int(center_x + side / 2 * exp)
    y_l = int(center_y - side / 2 * exp)
    y_r = int(center_y + side / 2 * exp)
    while x_l < 0 or y_l < 0 or x_r > 1920 or y_r > 1080:
        exp = random.uniform(1.05, 1.3) 
        x_l = int(center_x - side / 2 * exp)
        x_r = int(center_x + side / 2 * exp)
        y_l = int(center_y - side / 2 * exp)
        y_r = int(center_y + side / 2 * exp)   
     
    Crop = Dep[y_l:y_r, x_l:x_r]
    Mask_Crop = Mask[y_l:y_r, x_l:x_r]
    offset = np.array([[x_l, y_l, x_l, y_l]])
    bboxes = bboxes - offset
    
    # size 500 x 500
    Img = cv2.resize(Crop, (500, 500), interpolation=cv2.INTER_CUBIC)
    Mask_Img = cv2.resize(Mask_Crop, (500, 500), interpolation=cv2.INTER_CUBIC)
    np.save(os.path.join(Dirname, 'size_500_perfect.npy'), Img)
    Img_show = ((Img - Img.min()) / (Img.max() - Img.min()) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(Dirname, 'size_500_perfect_show.png'), Img_show)
    
    Img = add_noise(Img, Mask_Img)
    np.save(os.path.join(Dirname, 'size_500_noised.npy'), Img)
    Img_show = ((Img - Img.min()) / (Img.max() - Img.min()) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(Dirname, 'size_500_noised_show.png'), Img_show)

    bboxes_500 = bboxes * (500. / (side * exp))
    write_xml_single(os.path.join(Dirname, 'size_500.xml'), objects, bboxes_500)

    # size 300 x 300
    Img = cv2.resize(Crop, (300, 300), interpolation=cv2.INTER_CUBIC)
    Mask_Img = cv2.resize(Mask_Crop, (300, 300), interpolation=cv2.INTER_CUBIC)
    np.save(os.path.join(Dirname, 'size_300_perfect.npy'), Img)
    Img_show = ((Img - Img.min()) / (Img.max() - Img.min()) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(Dirname, 'size_300_perfect_show.png'), Img_show)

    Img = add_noise(Img, Mask_Img)
    np.save(os.path.join(Dirname, 'size_300_noised.npy'), Img)
    Img_show = ((Img - Img.min()) / (Img.max() - Img.min()) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(Dirname, 'size_300_noised_show.png'), Img_show)

    bboxes_300 = bboxes * (300. / (side * exp))
    write_xml_single(os.path.join(Dirname, 'size_300.xml'), objects, bboxes_300)

    # Write grasps
    write_grasps(os.path.join(Dirname, 'original.txt'), tasks, grasps, np.array([[0, 0]]), side * exp, side * exp)
    write_grasps(os.path.join(Dirname, 'size_300.txt'), tasks, grasps, np.array([[x_l, y_l]]), side * exp, 300)
    write_grasps(os.path.join(Dirname, 'size_500.txt'), tasks, grasps, np.array([[x_l, y_l]]), side * exp, 500)
    

def write_grasps(Filename, tasks, grasps, offset, side, target):
    box_len = 14. * target / 300
    os.mknod(Filename)
    with open(Filename, 'a') as f:
        for i, grasp in enumerate(grasps):
            for j, grasp_task in enumerate(grasp):
                if grasp_task.shape[0] == 0:
                   continue
                grasp_task = ((grasp_task - offset) * float(target) / float(side)).astype(np.int)
                grasp_v = grasp_task[0:grasp_task.shape[0]-1:2] - grasp_task[1:grasp_task.shape[0]:2]
                grasp_h = np.zeros_like(grasp_v)
                grasp_h[:, 0] = grasp_v[:, 1]
                grasp_h[:, 1] = -grasp_v[:, 0]
                norm = np.linalg.norm(grasp_h, axis = 1, keepdims=True)
                grasp_h_norm = grasp_h / norm

                for k in range(grasp_h.shape[0]):
                    point_0 = grasp_task[2 * k] - grasp_h_norm[k] * box_len / 2
                    f.write(str(point_0[0]) + ' ')
                    f.write(str(point_0[1]) + ' ')
                    point_1 = grasp_task[2 * k] + grasp_h_norm[k] * box_len / 2
                    f.write(str(point_1[0]) + ' ')
                    f.write(str(point_1[1]) + ' ')
                    point_2 = grasp_task[2 * k + 1] + grasp_h_norm[k] * box_len / 2
                    f.write(str(point_2[0]) + ' ')
                    f.write(str(point_2[1]) + ' ')
                    point_3 = grasp_task[2 * k + 1] - grasp_h_norm[k] * box_len / 2
                    f.write(str(point_3[0]) + ' ')
                    f.write(str(point_3[1]) + ' ')
                    f.write(str(i) + ' ')
                    f.write(tasks[i][j] + '\n')


if __name__ == '__main__':
    folder_path = os.path.dirname(os.path.abspath(__file__))
    for k, scenario in enumerate(scenario_names):
        Po = np.load(os.path.join(scenario_dir, scenario, 'Po.npy'))
        Or = np.load(os.path.join(scenario_dir, scenario, 'Or.npy'))
        back_idxes = [i for i in range(1, 7)]
        back_idx = random.choice(back_idxes)
        bpy.ops.wm.open_mainfile(filepath='backgrounds/background1.blend')
        # bpy.ops.wm.open_mainfile(filepath=(background%back_idx))
        f = open(os.path.join(scenario_dir, scenario, 'models.txt'))

        imported_objects = []
        grasps = []
        objects = []
        tasks = []
        for i, model in enumerate(f):
            imported_object = bpy.ops.import_scene.obj(filepath=os.path.join(model_dir, model.strip('\n'), 'model/model_normalized.obj'))
            objects.append(model.split('_')[0])
            imported_objects.append(bpy.context.selected_objects)
            for mesh in bpy.context.selected_objects:
                mesh.rotation_mode = 'QUATERNION'
                mesh.rotation_quaternion = [Or[i][3], Or[i][0], Or[i][1], Or[i][2]]
                mesh.location = Po[i]
                bpy.context.scene.update()
            if bpy.context.selected_objects[0].location.z < -1:
                imported_objects.pop(-1)
                objects.pop(-1)
                continue
            grasps_object = []
            task = []
            for grasps_task_object in os.listdir(os.path.join(model_dir, model.strip('\n'), 'annotations')):
                g_o = np.load(os.path.join(model_dir, model.strip('\n'), 'annotations', grasps_task_object))
                g_c = np.zeros_like(g_o)
                g_c[:, 0] = g_o[:, 0]
                g_c[:, 1] = g_o[:, 2]
                g_c[:, 2] = -g_o[:, 1]
                grasps_object.append(g_c)
                task.append(os.path.splitext(grasps_task_object)[0])
            grasps.append(grasps_object)
            tasks.append(task)

        f.close()
        scene = bpy.context.scene
        cam_ob = bpy.context.scene.camera
        bpy.context.scene.render.resolution_percentage = 100
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(scenario_dir, scenario, 'scenario.blend'))
        bboxes, grasps_new = get_bboxes_and_grasps2(scene, cam_ob, imported_objects, grasps, os.path.join(folder_path, 'tmp'))
        bpy.ops.wm.open_mainfile(filepath=os.path.join(scenario_dir, scenario, 'scenario.blend'))
        os.mkdir(os.path.join(datasets_dir, 'depth_%06d'%(start_idx + k)))
        get_depth(os.path.join(datasets_dir, 'depth_%06d'%(start_idx + k)))

        bboxes = np.array(bboxes)
        write_xml_and_depth_and_grasps(os.path.join(datasets_dir, 'depth_%06d'%(start_idx + k)), objects, tasks, bboxes.astype(np.int), grasps_new)

        # write_xml(os.path.join(datasets_dir, 'depth_%06d'%(start_idx + k)) + '/%06d.xml' % (start_idx + k), objects, bboxes.astype(np.int))
        # np.save(os.path.join(datasets_dir, 'depth_%06d'%(start_idx + k), 'bboxes.npy') , bboxes.astype(np.int))
        # write_grasps(os.path.join(datasets_dir, 'depth_%06d'%(start_idx + k)) + '/%06d.txt' % (start_idx + k), tasks, grasps_new)

        # for i, grasp in enumerate(grasps_new):
        #     for j, grasp_t in enumerate(grasp):
        #         np.save(os.path.join(datasets_dir, 'depth_%06d'%(start_idx + k), 'grasp%d%d.npy'%(i, j)), grasp_t)

