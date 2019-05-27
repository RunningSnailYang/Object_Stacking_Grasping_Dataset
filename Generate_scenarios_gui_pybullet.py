import argparse
import os
import random
import time

import numpy as np
import pybullet as p
import pybullet_data

def Extract_Object():
    models = os.listdir('Datasets/3D_models')
    models_use = models
    '''
    models_use = []
    for model in models:
        if 'spatula' in model:
            models_use.append(model)
    '''
    models_num = random.randint(3, 6)
    models_sample = random.sample(models_use, models_num)
    return models_sample

if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='The number of scenarios you want to generate')
    parser.add_argument('--num_scenarios', help='The number of scenarios you want to generate',
                        default=1000, type=int)
    parser.add_argument('--gui', help='Whether to use GUI mode', action='store_true')
    args = parser.parse_args()

    if os.path.exists('Datasets/Scenarios'):
        start_idx = len(os.listdir('Datasets/Scenarios'))
    else:
        os.mkdir('Datasets/Scenarios')
        start_idx = 0

    # Generate scenarios
    for i in range(args.num_scenarios):
        dir_name = os.path.join('Datasets', 'Scenarios', 'Scenario_%06d'%(i + start_idx))
        models_name = os.path.join(dir_name, 'models.txt')
        os.mkdir(dir_name)
        os.mknod(models_name)
        f = open(models_name, 'w')
        models = Extract_Object()
        Po = []
        Or = []
        if args.gui:
            physicsClient = p.connect(p.GUI)
        else:
            physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-10)
        planeId = p.loadURDF("plane1.urdf")
        V_ID = []
        C_ID = []
        O_ID = []

        for model in models:
            f.write(model + '\n')
            V_ID.append(p.createVisualShape(shapeType = p.GEOM_MESH, fileName = os.path.join('Datasets', '3D_models', model, 'model', 'model_normalized.obj')))
            C_ID.append(p.createCollisionShape(shapeType = p.GEOM_MESH, fileName = os.path.join('Datasets', '3D_models', model, 'model', 'model_normalized.obj')))

        for i in range(len(V_ID)):
            ori = [random.random() for j in range(4)]
            ori.sort()
            ori.append(1)
            ori_e = [ori[1]-ori[0], ori[2]-ori[1], ori[3]-ori[2], ori[4]-ori[3]]
            O_ID.append(p.createMultiBody(baseMass=1,baseInertialFramePosition=[0,0,0],baseCollisionShapeIndex=C_ID[i], baseVisualShapeIndex = V_ID[i], basePosition = [0,0,1], baseOrientation = ori_e))
            p.setGravity(0,0,-10)
            p.setRealTimeSimulation(1)
            # p.stepSimulation()
            time.sleep(1)
        time.sleep(3)
        for i in range(len(V_ID)):
            f1 , f2 = p.getBasePositionAndOrientation(O_ID[i])
            Po.append(f1)
            Or.append(f2)
        Po = np.array(Po)
        Or = np.array(Or)
        np.save(os.path.join(dir_name, 'Po.npy'), Po)
        np.save(os.path.join(dir_name, 'Or.npy'), Or)
        f.close()
        p.disconnect()
    
