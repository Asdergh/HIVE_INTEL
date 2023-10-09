import numpy as np
import pandas as pd
import pyvista as pv


view_3d = pv.Plotter()


cone_one = pv.Cone(np.zeros(3), np.random.normal(4.5, 1.23, size=3), radius=0.12)
cone_two = pv.Cone(np.zeros(3), np.random.normal(4.5, 1.23, size=3), radius=0.12)
cone_free = pv.Cone(np.zeros(3), np.random.normal(4.5, 1.23, size=3), radius=0.12)
cones_info = {}

for cone_number in range(300):
    
    new_cone_cores = np.array([np.zeros(3), np.random.normal(12.3, 5.6, 3)])
    new_cone = pv.Cone(new_cone_cores[0], new_cone_cores[1])
    view_3d.add_mesh(new_cone, style="wireframe", color="red")

    cones_info[f"cone_number: {cone_number}"] = {
        "start_positoin: ": new_cone_cores[0],
        "end_position: ": new_cone_cores[1]
    }

print(cones_info)
print(view_3d.meshes)
for frame in range(100):
    view_3d.update_coordinates(np.random.normal(12.6, 5.6, 3))





view_3d.background_color = "gray"
view_3d.show()


