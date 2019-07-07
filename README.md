# Object Stacking Grasping Dataset
This is the code to generate Object Stacking Grasping Dataset(OSGD)

## Generate Object Stacking Scenarios
To generate the object stacking scenarios, you can run these commands:
* Generate scenarios without GUI:  
```
python Generate_scenarios_pybullet.py
```
* Generate scenarios with GUI:  
```
python Generate_scenarios_pybullet.py --gui
```

## Generate Depth Images and Annotations
After the scenarios have been generated, you can generate the final depth maps and their annotations by running this command:
```
blender --background --python Generate_final_datasets.py
```

