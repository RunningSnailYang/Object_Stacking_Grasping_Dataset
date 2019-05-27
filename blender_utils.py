import os
import bpy
import sys
import cv2
import numpy as np
import time

def set_layer(obj, layer_idx):
    """ Move an object to a particular layer """
    # Set the target layer to True first because an object must always be on
    # at least one layer.
    obj.layers[layer_idx] = True
    for i in range(len(obj.layers)):
        obj.layers[i] = (i == layer_idx)

def Check_Models_Usability(bk_dir, ip_dir, op_dir, category):
    '''
    This function checks if the models can be rendered as white and black
    bk_dir: The stored directory of the background
    ip_dir: The input directory which the models are stored
    op_dir: The output directory to save rendered results
    category: The category of the models to check, such as knife, pliers
    '''
    # Get the folder names of models we need
    models = []
    for model in os.listdir(ip_dir):
        if category in model:
            models.append(os.path.join(ip_dir, model, 'model', 'model_normalized.obj'))

    for model in models:
        Check_SingleModel_Usability(bk_dir, model, op_dir)

    
def Check_SingleModel_Usability(bk_dir, model, op_dir):
    '''
    This function checks if the models can be rendered as white and black
    bk_dir: The stored directory of the background
    ip_dir: The input directory which the models are stored
    op_dir: The output directory to save rendered results
    category: The category of the models to check, such as knife, pliers
    '''
    colors = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
    render_args = bpy.context.scene.render
    render_args.engine = 'BLENDER_RENDER'
    render_args.use_antialiasing = False
    bpy.context.scene.update()

    bpy.ops.wm.open_mainfile(filepath = os.path.join(bk_dir, 'background.blend'))
    imported_object = bpy.ops.import_scene.obj(filepath = model)
    selected_objects = bpy.context.selected_objects
    set_layer(bpy.data.objects['Plane'], 2)
    set_layer(bpy.data.objects['Lamp'], 2)

    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for n in tree.nodes:
        tree.nodes.remove(n)
    rl = tree.nodes.new('CompositorNodeRLayers')  
    vl = tree.nodes.new('CompositorNodeOutputFile')  # Node to output depth
    links.new(rl.outputs[0], vl.inputs[0])
    vl.base_path = op_dir

    for c, color in enumerate(colors):
        for i, obj in enumerate(selected_objects):
            bpy.ops.material.new()
            mat = bpy.data.materials['Material']
            mat.name = 'Material_%d' % i
            mat.diffuse_color = color
            mat.use_shadeless = True
            obj.data.materials[0] = mat
        bpy.ops.render.render(write_still=True)
        file_num = len(os.listdir(op_dir))
        name = model.split('/')[-3]
        os.rename(os.path.join(op_dir, 'Image0001.png'), os.path.join(op_dir, 'Img'+ name + '%06d.png'% c))

def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))

def grasps_nms(grasps, threshold):
    '''
    Filiter redundant grasps 
    Parameters:
    grasps: The grasps in world coordinate system. shape: [grasps_num x 2, 4]
    threshold: The distance threshold which should be suppressed
    '''
    points_num = grasps.shape[0] # Grasps' points number
    grasps_v = (grasps[0:points_num - 1:2] - grasps[1:points_num:2])[:, :3] # Grasps' orientations
    grasps_c = (grasps[0:points_num - 1:2] + grasps[1:points_num:2])[:, :3] / 2 # Grasps' centers
    # Consine of the included angle between Grasps' orientations and (0, 0, 1) in world coordinates system
    cos_grasps = np.dot(grasps_v[:, :3], np.array([0, 0, 1])) / np.linalg.norm(grasps_v[:, :3], axis = 1)
    cos_grasps_abs = np.fabs(cos_grasps)
    # Grasps which have less cos_grasps_abs should be retained
    # Suppress them by distances
    keep_idxes = []
    grasps_keep = []
    sorted_idxes = np.argsort(cos_grasps_abs)
    c = 0
    while sorted_idxes.shape[0] > 0:
        c += 1
        idx = sorted_idxes[0]
        keep_idxes.append(idx)
        grasps_keep.append(grasps[idx *2])
        grasps_keep.append(grasps[idx *2 + 1])
        sorted_idxes = sorted_idxes[1:]
        if sorted_idxes.shape[0] > 0:
            distances = np.linalg.norm((grasps_c[sorted_idxes, :3] - grasps_c[idx, :3]), axis = 1)
            sorted_idxes = sorted_idxes[distances > threshold]
    grasps = np.array(grasps_keep)
    return grasps

def get_bound_and_grasps(scene, cam_ob, objs, grasps_task):
    
    '''Get the parameters of camera'''
    # Get Affine Transformation Matrix from world coordinates to camera coordinates.
    # mat: shape: [4, 4], content: [R|t]
    mat = cam_ob.matrix_world.normalized().inverted()
    camera = cam_ob.data # cam_ob is an object contain camera, cam_ob.data return the camera
    frame = [-v for v in camera.view_frame(scene=scene)[:3]] # Return 4 corners for the cameras frame
    camera_persp = camera.type != 'ORTHO' # Judge whether the camera type is perspective or not

    grasps_task_covert = []
    ma = objs[0].matrix_world
    for grasps in grasps_task:
        '''Get the grasps in world coordinates'''
        grasps_num = grasps.shape[0]
        grasps = np.concatenate((grasps, np.ones((grasps_num, 1))), axis = 1)
        grasps = np.dot(ma, grasps.T).T

        '''Filter the grasps which are can't be executed'''
        grasps_keep = []
        for i in range(int(grasps.shape[0] / 2)):
            v_grasp = grasps[i * 2 + 1] - grasps[i * 2]
            cos_v = np.dot(v_grasp[0:3], np.array([0, 0, 1])) / (np.linalg.norm(v_grasp) * 1)
            if cos_v >= -0.708 and cos_v <= 0.708:
                grasps_keep.append(grasps[i * 2])
                grasps_keep.append(grasps[i * 2 + 1])
        grasps = np.array(grasps_keep)
    
        '''Filiter redundant grasps'''
        if grasps.shape[0] > 0:
            grasps = grasps_nms(grasps, 0.05)

            '''Get the grasps in camera coordinates'''
            grasps = np.dot(mat, grasps.T).T
            '''Get the grasps' relative coordinates in the final image'''
            for i in range(grasps.shape[0]):
                z = -grasps[i, 2]
                frame = [(v / (v.z / z)) for v in frame]
                min_x, max_x = frame[0].x, frame[2].x
                min_y, max_y = frame[0].y, frame[2].y
                grasps[i][0] = (grasps[i][0] - min_x) / (max_x - min_x) 
                grasps[i][1] = 1 - (grasps[i][1] - min_y) / (max_y - min_y) 
            grasps_task_covert.append(grasps[:, :2])
        else:
            grasps_task_covert.append(grasps)

    '''Convert each mesh to camera system coordinates'''
    meshes = [] # List to restore all meshes of a real object
    for obj in objs: # Transform each object from object coordinate system to camera system
        mesh = obj.to_mesh(scene, True, 'PREVIEW')  # Get the mesh of an object
        # mesh.transform(mat) equals to mat x points.T
        mesh.transform(obj.matrix_world) # From object coordinate system to world coordinate system
        mesh.transform(mat) # From world coordinate system to camera coordinate system
        meshes.append(mesh)

    '''Get the relative coordinates of all vertices in the final image'''
    lx = [] # All relative x coordinates of these meshes
    ly = [] # All relative y coordinates of these meshes
    for mesh in meshes: # For each mesh, calculate the relative x, y coordinates of its vertices
        for v in mesh.vertices: # Calculate coordinates of each vertices
            co_local = v.co # Get the coordinate of a vertice
            z = -co_local.z
            if camera_persp:
                if z == 0.0:
                    lx.append(0.5)
                    ly.append(0.5)
                else:
                    frame = [(v / (v.z / z)) for v in frame] # Get the frame parallel to the camera and where v locates in

            min_x, max_x = frame[0].x, frame[2].x
            min_y, max_y = frame[0].y, frame[2].y
            # According to principle of similar triangles, relative coordinate in this frame equals its in the final image
            x = (co_local.x - min_x) / (max_x - min_x) 
            y = (co_local.y - min_y) / (max_y - min_y)

            lx.append(x)
            ly.append(y)

    '''Get the coordinates' maximum and minimum value of these vertices'''
    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac
    # Because the y axis of camera and image are in opposite direction, it is necessary to convert it.
    return min_x, 1.0 - max_y, max_x, 1.0 - min_y, dim_x, dim_y, grasps_task_covert

def get_bboxes_and_grasps(scene, cam_ob, objs, grasps, tmp_dir):
    render_args = bpy.context.scene.render

    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing

    render_args.engine = 'BLENDER_RENDER'
    render_args.use_antialiasing = False

    set_layer(bpy.data.objects['Plane'], 2)
    set_layer(bpy.data.objects['Lamp'], 2)

    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for n in tree.nodes:
        tree.nodes.remove(n)
    rl = tree.nodes.new('CompositorNodeRLayers')  
    vl = tree.nodes.new('CompositorNodeOutputFile')  # Node to output depth
    links.new(rl.outputs[0], vl.inputs[0])
    vl.base_path = tmp_dir

    # Create new materials for each parts of these objects
    old_materials = []
    mat_idx = 0
    for obj in objs:
        for part in obj:
            old_materials.append(part.data.materials[0])
            bpy.ops.material.new()
            mat = bpy.data.materials['Material']
            mat.name = 'Material_%d' % mat_idx
            mat.diffuse_color = [0.0, 0.0, 0.0]
            mat.use_shadeless = True
            part.data.materials[0] = mat
            mat_idx += 1

    for i, obj in enumerate(objs):
        for part in obj:
            bpy.ops.material.new()
            mat = bpy.data.materials['Material']
            mat.name = 'Material_%d' % mat_idx
            mat.diffuse_color = [1.0, 1.0, 1.0]
            mat.use_shadeless = True
            part.data.materials[0] = mat
            mat_idx += 1
            # part.data.materials[0].diffuse_color = [1.0, 1.0, 1.0]
        bpy.ops.render.render(write_still=True)
        os.rename(tmp_dir + '/Image0001.png', tmp_dir + '/ff%d.png'%i)
        for part in obj:
            bpy.ops.material.new()
            mat = bpy.data.materials['Material']
            mat.name = 'Material_%d' % mat_idx
            mat.diffuse_color = [0.0, 0.0, 0.0]
            mat.use_shadeless = True
            part.data.materials[0] = mat
            mat_idx += 1
    print(objs)

def get_bboxes_and_grasps2(scene, cam_ob, objs, grasps, tmp_dir):
    dir_num = len(os.listdir(tmp_dir))
    dir_name = os.path.join(tmp_dir, 're%d'%dir_num)
    os.mkdir(dir_name)
    render_args = bpy.context.scene.render
    render_args.resolution_percentage = 100
    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing

    render_args.engine = 'BLENDER_RENDER'
    render_args.use_antialiasing = True

    set_layer(bpy.data.objects['Plane'], 2)
    set_layer(bpy.data.objects['Lamp'], 2)

    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Create new materials for each parts of these objects
    old_materials = []
    for mat in bpy.data.materials.keys():
        bpy.data.materials.remove(bpy.data.materials[mat])

    for i in range(len(objs)):
        #while True:

        for n in tree.nodes:
            tree.nodes.remove(n)
        rl = tree.nodes.new('CompositorNodeRLayers')  
        vl = tree.nodes.new('CompositorNodeOutputFile')  # Node to output depth
        links.new(rl.outputs[0], vl.inputs[0])
        vl.base_path = dir_name

        mat_idx = 0
        for j, obj in enumerate(objs):
            for part in obj:
                old_materials.append(part.data.materials[0])
                bpy.ops.material.new()
                mat = bpy.data.materials['Material']
                mat.name = 'Material_%d' % mat_idx
                if j == i:
                    mat.diffuse_color = [1.0, 1.0, 1.0]
                else:
                    mat.diffuse_color = [0.1, 0.1, 0.1]
                mat.use_shadeless = True
                part.data.materials[0] = mat
                mat_idx += 1
        bpy.ops.render.render(write_still=True)
        #if Img.min() != 0:
        #    break
        os.rename(dir_name + '/Image0001.png', dir_name + '/ff%d.png'%i)
        for mat in bpy.data.materials.keys():
            bpy.data.materials.remove(bpy.data.materials[mat])
    
    bboxes = []
    grasps_new = []
    for i in range(len(objs)):
        Annotations = get_bound_and_grasps(scene, cam_ob, objs[i], grasps[i])
        bbox = list(Annotations[:4])
        grasp_task = Annotations[6]
        bbox[0] = int(bbox[0] * 1920)
        bbox[1] = int(bbox[1] * 1080)
        bbox[2] = int(bbox[2] * 1920)
        bbox[3] = int(bbox[3] * 1080)

        Img = cv2.imread(dir_name + '/ff%d.png'%i)
        Crop = Img[bbox[1]:bbox[3], bbox[0]:bbox[2], 0]
        Img = Img[:, :, 0]
        Mask = Crop < 150
        # Mask = np.concatenate((Mask[:, :, np.newaxis], Mask[:, :, np.newaxis]), axis = 2)
        Coordinates = np.ones((Crop.shape[0], Crop.shape[1], 2), np.int)
        Coordinates[:, :, 0] = Coordinates[:, :, 0].cumsum(axis = 0) - 1
        Coordinates[:, :, 1] = Coordinates[:, :, 1].cumsum(axis = 1) - 1
        x_max = (Coordinates[:, :, 1] - Mask * 260).max() + bbox[0]
        x_min = (Coordinates[:, :, 1] + Mask * 260).min() + bbox[0]
        y_max = (Coordinates[:, :, 0] - Mask * 260).max() + bbox[1]
        y_min = (Coordinates[:, :, 0] + Mask * 260).min() + bbox[1]

        grasp_task_new = []
        for grasp in grasp_task:
            grasp_new = []
            if len(grasp_task) == 0:
                grasp_task_new.append(grasp_new.astype(np.int))
                continue
            for j in range(int(grasp.shape[0] / 2)):
                grasp_p1 = grasp[2 * j] * np.array([1920, 1080])
                grasp_p2 = grasp[2 * j + 1] * np.array([1920, 1080])
                center = (grasp_p1 + grasp_p2) / 2 
                center_x = int(center[0])
                center_y = int(center[1])

                len_x = abs(grasp_p1[0] - grasp_p2[0])
                len_y = abs(grasp_p1[1] - grasp_p2[1])
                inter_num = int(max(len_x, len_y))
                grasp_line = np.array([np.linspace(grasp_p1[0], grasp_p2[0], inter_num),
                                           np.linspace(grasp_p1[1], grasp_p2[1], inter_num)]).astype(np.int)
                line_l = grasp_line[:, :(inter_num / 3)]
                line_m = grasp_line[:, (inter_num / 3):(2 * inter_num / 3)]
                line_r = grasp_line[:, (2 * inter_num / 3):]
      
                line_l2 = grasp_line[:, :(inter_num / 2)]
                line_r2 = grasp_line[:, (inter_num / 2):]
                if Img[center_y, center_x] > 150:
                    condition = (((Img[line_l2[1], line_l2[0]] < 91) * (Img[line_l2[1], line_l2[0]] > 87)).sum() / float(line_l2.shape[1]) > 0.5 and \
                                ((1 - (Img[line_r2[1], line_r2[0]] < 91) * (Img[line_r2[1], line_r2[0]] > 87)).sum()) / float(line_r2.shape[1]) > 0.5) or \
                                (((Img[line_r2[1], line_r2[0]] < 91) * (Img[line_r2[1], line_r2[0]] > 87)).sum() / float(line_r2.shape[1]) > 0.5 and \
                                ((1 - (Img[line_l2[1], line_l2[0]] < 91) * (Img[line_l2[1], line_l2[0]] > 87)).sum()) / float(line_l2.shape[1]) > 0.5)
                    if not condition:
                        grasp_new.append(grasp[2 * j])
                        grasp_new.append(grasp[2 * j + 1])
                else:
                    len_x = abs(grasp_p1[0] - grasp_p2[0])
                    len_y = abs(grasp_p1[1] - grasp_p2[1])
                    inter_num = int(max(len_x, len_y))
                    grasp_line = np.array([np.linspace(grasp_p1[0], grasp_p2[0], inter_num),
                                           np.linspace(grasp_p1[1], grasp_p2[1], inter_num)]).astype(np.int)
                    line_l = grasp_line[:, :(inter_num / 3)]
                    line_m = grasp_line[:, (inter_num / 3):(2 * inter_num / 3)]
                    line_r = grasp_line[:, (2 * inter_num / 3):]
                    condition = ((Img[grasp_line[1], grasp_line[0]] < 91) * (Img[grasp_line[1], grasp_line[0]] > 87)).sum() / float(grasp_line.shape[1]) < 0.04 and \
                                (Img[line_l[1], line_l[0]] == 255).sum() > 5 and (Img[line_r[1], line_r[0]] == 255).sum() > 5
                    if condition:
                        grasp_new.append(grasp[2 * j])
                        grasp_new.append(grasp[2 * j + 1])
                    
            grasp_new = np.array(grasp_new)
            if grasp_new.shape[0] > 0:
                grasp_new[:, 0] = grasp_new[:, 0] * 1920
                grasp_new[:, 1] = grasp_new[:, 1] * 1080
            grasp_task_new.append(grasp_new.astype(np.int))
        bboxes.append([x_min, y_min, x_max, y_max])
        grasps_new.append(grasp_task_new)

    set_layer(bpy.data.objects['Plane'], 0)
    set_layer(bpy.data.objects['Lamp'], 0)
    return bboxes, grasps_new

def get_depth(output_dir):

    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for n in tree.nodes:
        tree.nodes.remove(n)
    rl = tree.nodes.new('CompositorNodeRLayers')      

    vl = tree.nodes.new('CompositorNodeOutputFile')  # Node to output depth
    vl.format.file_format = 'OPEN_EXR' # Change the format of the output file
    vi = tree.nodes.new('CompositorNodeOutputFile')  # Node to output image
    links.new(rl.outputs[0], vi.inputs[0])  # link Render image to the output node vi
    links.new(rl.outputs[2], vl.inputs[0])  # link Render depth to the output node vl
    vl.base_path = output_dir # Set the output folder of the output node
    vi.base_path = output_dir
    #render
    bpy.context.scene.render.resolution_percentage = 100 # make sure scene height and width are ok (edit)
    bpy.ops.render.render(write_still=True)
    Dep = cv2.imread(os.path.join(output_dir, 'Image0001.exr'), cv2.IMREAD_UNCHANGED)
    Dep = Dep[:, :, 0]
    Dep = (Dep - Dep.min()) / (Dep.max() - Dep.min()) * 255.
    Dep = Dep.astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, 'Image0001_dep.png'), Dep)
       


        
