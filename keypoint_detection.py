




# Import packages
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scipy') 
import os
import sys
import vedo
import time
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util.numpy_support import vtk_to_numpy 
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist
import multiprocessing
import itertools
from scipy import spatial

def gaussian(x, mean, sigma):
    """ Subfunction for gaussian filter"""
    """
    Input: array with values, mean of gaussian, standard deviation of gaussian
    Output: array of filtered values
    """
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mean) ** 2) / (sigma ** 2)
    return coefficient * np.exp(exponent)


def boundaries(
        self,
        boundary_edges=True,
        manifold_edges=False,
        non_manifold_edges=False,
        feature_angle=None,
        return_point_ids=False,
        return_cell_ids=False,
        cell_edge=False,
    ):
        """
        Source: https://vedo.embl.es/autodocs/content/vedo/vedo.html
        Return the boundary lines of an input mesh.
        Check also `vedo.base.BaseActor.mark_boundaries()` method.

        Arguments:
            boundary_edges : (bool)
                Turn on/off the extraction of boundary edges.
            manifold_edges : (bool)
                Turn on/off the extraction of manifold edges.
            non_manifold_edges : (bool)
                Turn on/off the extraction of non-manifold edges.
            feature_angle : (bool)
                Specify the min angle btw 2 faces for extracting edges.
            return_point_ids : (bool)
                return a numpy array of point indices
            return_cell_ids : (bool)
                return a numpy array of cell indices
            cell_edge : (bool)
                set to `True` if a cell need to share an edge with 
                the boundary line, or `False` if a single vertex is enough

        Examples:
            - [boundaries.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/boundaries.py)

            ![](https://vedo.embl.es/images/basic/boundaries.png)
        """
        fe = vtk.vtkFeatureEdges()
        fe.SetBoundaryEdges(boundary_edges)
        fe.SetNonManifoldEdges(non_manifold_edges)
        fe.SetManifoldEdges(manifold_edges)
        # fe.SetPassLines(True) # vtk9.2
        fe.ColoringOff()
        fe.SetFeatureEdges(False)
        if feature_angle is not None:
            fe.SetFeatureEdges(True)
            fe.SetFeatureAngle(feature_angle)

        if return_point_ids or return_cell_ids:
            idf = vtk.vtkIdFilter()
            idf.SetInputData(self.polydata())
            idf.SetPointIdsArrayName("BoundaryIds")
            idf.SetPointIds(True)
            idf.Update()

            fe.SetInputData(idf.GetOutput())
            fe.Update()

            vid = fe.GetOutput().GetPointData().GetArray("BoundaryIds")
            npid = vtk_to_numpy(vid).astype(int)

            if return_point_ids:
                return npid

            if return_cell_ids:
                n = 1 if cell_edge else 0
                inface = []
                for i, face in enumerate(self.faces()):
                    # isin = np.any([vtx in npid for vtx in face])
                    isin = 0
                    for vtx in face:
                        isin += int(vtx in npid)
                        if isin > n:
                            break
                    if isin > n:
                        inface.append(i)
                return np.array(inface).astype(int)

            return self

        else:

            fe.SetInputData(self.polydata())
            fe.Update()
            msh = Mesh(fe.GetOutput(), c="p").lw(5).lighting("off")

            msh.pipeline = OperationNode(
                "boundaries", 
                parents=[self], 
                shape="octagon",
                comment=f"#pts {msh._data.GetNumberOfPoints()}",
            )
            return msh
            
            
def get_connected(I, mesh):
    """Get connected vertices to a list of vertices"""
    """ 
    Input: index of point to find connections with, mesh to find connected in
    Output: List of connected vertices
    """
    connected = [mesh.connectedVertices(num, returnIds=True) for num in I]
    return connected
    
def find_subset(I, mesh, ring=3):
    """Find connected vertices in a k-ring neighborhood"""
    """
    Input: Center vertex, mesh, and size of neighborhood
    Output: List of main vertices and nested list of connected vertices
    """
    # Initialize
    idx1 = []
    idx2 = []
    cumul = {I}

    # Iterate through rings
    for _ in range(ring):
        # Collect connected vertices
        con = get_connected(cumul, mesh)
        idx1.extend(cumul)
        idx2.extend(con)
        cumul.update(itertools.chain.from_iterable(con))

    # Remove redundancy
    remove = set()
    mem = set()
    for i, id_x in enumerate(idx1):
        if id_x in mem:
            remove.add(i)
        else:
            mem.add(id_x)
    idx1 = [i for j, i in enumerate(idx1) if j not in remove]
    idx2 = [i for j, i in enumerate(idx2) if j not in remove]

    return idx1, idx2
    
    
#Function for calculating curvatures
def compute_curvature(mesh, method=0):
    """ Function for calculating the curvature """
    """
    Input: Mesh-cut to calculate curvature of, method 0 is for gaussian curvature, 1 is for mean curvature 
    Output: the curvature array representing the curvature of the mesh section
    """
    curve = vtk.vtkCurvatures()
    curve.SetInputData(mesh._data)
    curve.SetCurvatureType(method)
    curve.Update()
    dataObject = dsa.WrapDataObject(curve.GetOutput())
    if method == 0:
        meth_name = "Gauss_Curvature"
    elif method == 1:
        meth_name = "Mean_Curvature"
    else:
        #print("Unknown curvature method")
        sys.exit()
        
        
    curvature_array = dataObject.PointData[meth_name] # output array.
    curvature_array = vtk_to_numpy(curvature_array)

    return curvature_array
    
def points_in_radius(mesh,point,r=1):
    """ Function for selecting points within a certain radius"""
    """
    Input: radius, mesh and start point
    Output: list of point IDS
    """
    point = np.array(point)
    mesh_points = np.array(mesh.points())
    tree = spatial.KDTree(mesh_points)
    keep_index = tree.query_ball_point(point, r)
    return keep_index


def principal_curvature(mesh):
    """Function for calculating the principal curvatures from the Gaussian and the mean"""
    """
    Input: Mesh to calculate principal curvature for
    Output: Principal curvature (min and max), mean curvature, and Gaussian curvature as arrays
    """

    # Calculate mean and Gaussian curvatures
    mean_curv = compute_curvature(mesh, method=1)
    gaus_curv = compute_curvature(mesh, method=0)

    # Adjust mean curvature and Gaussian curvature based on conditions
    valid_indices = np.logical_and(mean_curv > -1, mean_curv < 1)
    mean_curv = np.where(valid_indices, mean_curv, 0)
    gaus_curv = np.where(mean_curv ** 2 >= gaus_curv, gaus_curv, mean_curv ** 2 - 0.01)

    # Calculate principal curvatures
    min_curv = mean_curv - np.sqrt(mean_curv ** 2 - gaus_curv)
    max_curv = mean_curv + np.sqrt(mean_curv ** 2 - gaus_curv)

    return min_curv, max_curv, mean_curv, gaus_curv
    
    
def sort_index(lst, rev=True):
    """
    Function for getting sorted index order
    """
    indices = np.argsort(lst)[::-1 if rev else 1]
    return indices
    
   
    
def check_keypoint(p, collected_subset_min, collected_subset_max,contrast_threshold, bounds, ref_mesh):
    # Ignore low contrast keypoints
    if (np.max(collected_subset_min) - np.min(collected_subset_min)) > contrast_threshold:
        # Ignore points close to boundaries
        for bound_id in bounds:
            q = ref_mesh.points()[bound_id]
            if np.linalg.norm(p - q) < 1:
                return False
        return True
    return False
    
    
def keypoint_detection(mesh, name = "test", res=20, returnIdx = False, returnPts = True, inspection = True, output="./"):
    """ Function for detecting robust keypoints"""
    """
    Input: mesh to find keypoints on, a name for output files, output path, 
    Output: keypoints
    """
    
    # Collection of possible keypoints, initialization
    keypoints = []
    
    # Decimate for reproducibility
    #print("Decimating/subdividing")
    teeth = mesh.clone()
    area = teeth.area()
    n = len(teeth.points())
    n_a = round(res*area)
    frac = n_a/n
    #print("Aimed resolution: ",round(n_a/area,3))
    teeth = teeth.decimate(fraction=frac ,method='quadratic')
    
    #If the decimation doesn't work, try again with a different fraction
    if len(mesh.faces())*(frac+0.5) < len(teeth.faces()):
        #print("Decimating error. Retrying...")
        frac = frac+0.2
        teeth = mesh.clone()
        teeth.decimate(fraction=frac)
        if len(mesh.faces())*(frac+0.5) < len(teeth.faces()):
            #print("Could not sucessfully decimate mesh. Try to do so manually.")
            sys.exit(1)
    #print("True resolution: ",round(len(teeth.points())/area,3))
    
    # Work with a mesh clone
    submesh = teeth.clone()

    # Apply Gaussian smoothing to deal with Difference of Curvature (DoC) - Initialization
    #print("Preparing DoGs")
    DoG = []
    dog_meshes = [submesh.clone() for _ in range(6)]
    ref_mesh = submesh.clone()
    total = len(ref_mesh.points())
    sigmas = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

    # Iterate through points in mesh
    for i, p in enumerate(ref_mesh.points()):
        #print("    ", round(((i + 1) / total) * 100, 2), " %   ", end="\r")

        # Find distances to center point for all points in the neighborhood
        connect_idx = points_in_radius(ref_mesh, p, r=4)
        distances = np.linalg.norm(ref_mesh.points() - p, axis=1)[connect_idx]

        # Use Gaussian filter to assign weights to each point dependent on distance to center point
        weights = np.array([gaussian(distances, 0, sigma) for sigma in sigmas])

        # Recalculate vertex position for each of the smoothings
        smoothed_points = [np.average(ref_mesh.points()[connect_idx], weights=weight, axis=0) for weight in weights]

        # Update dog_mesh points
        for j, dog_mesh in enumerate(dog_meshes):
            dog_mesh.points()[i] = smoothed_points[j]

    # Collect the smoothed meshes
    DoG.extend(dog_meshes)
    if inspection == True:
        for i in range(len(sigmas)):
            vedo.write(DoG[i],os.path.join(output, f"smoothmesh_{name}_{i}_.vtp"))

                
    # Find curvature values
    dog_min = []
    dog_max = []
    dog_mean = []
    dog_gaus = []
    for dog in DoG:
        dog_mesh = dog.clone()
        min_curv, max_curv, mean_curv, gaus_curv = principal_curvature(dog_mesh)
        dog_min.append(min_curv)
        dog_max.append(max_curv)
        dog_mean.append(mean_curv)
        dog_gaus.append(gaus_curv)

    # find DoC differences
    #print("    Working with DoG pyramids                                     ")
    D1_min = np.subtract(dog_min[0],dog_min[1])
    D2_min = np.subtract(dog_min[1],dog_min[2])
    D3_min = np.subtract(dog_min[2],dog_min[3])
    D4_min = np.subtract(dog_min[3],dog_min[4])
    D5_min = np.subtract(dog_min[4],dog_min[5])
    D1_max = np.subtract(dog_max[0],dog_max[1])
    D2_max = np.subtract(dog_max[1],dog_max[2])
    D3_max = np.subtract(dog_max[2],dog_max[3])
    D4_max = np.subtract(dog_max[3],dog_max[4])
    D5_max = np.subtract(dog_max[4],dog_max[5])
    
    # Initialize keypoint detection
    bounds = boundaries(ref_mesh,return_point_ids=True)
    keypoints_id = []
    contrast_threshold = 0.00001   #0.00001
    

    # Iterate through points in mesh
    for I, p in enumerate(ref_mesh.points()):
        #print("    ", round(((I + 1) / total) * 100, 2), " %   ", end="\r")
        kp = False

        # Compare closest neighborhood
        main_idx, connect_idx = find_subset(I, ref_mesh, ring=1)
        connect_idx = list(set(sum(connect_idx, [])))
    
        # Collect min/max subsets
        collected_subset_min = np.concatenate((D1_min[connect_idx], D2_min[connect_idx], D3_min[connect_idx]), axis=0)
        collected_subset_max = np.concatenate((D1_max[connect_idx], D2_max[connect_idx], D3_max[connect_idx]), axis=0)
        #collected_subset_min = np.concatenate((D1_min[connect_idx], D2_min[connect_idx], D4_min[connect_idx],D5_min[connect_idx],D3_min[connect_idx]), axis=0)
        #collected_subset_max = np.concatenate((D1_max[connect_idx], D2_max[connect_idx], D4_max[connect_idx],D5_max[connect_idx],D3_max[connect_idx]), axis=0)
    
        if D2_min[I] == np.min(collected_subset_min) or D2_min[I] == np.max(collected_subset_min):
            if check_keypoint(p, collected_subset_min, collected_subset_max, contrast_threshold, bounds, ref_mesh):
                keypoints_id.append(I)

        elif D3_min[I] == np.min(collected_subset_min) or D3_min[I] == np.max(collected_subset_min):
            if check_keypoint(p, collected_subset_min, collected_subset_max, contrast_threshold, bounds, ref_mesh):
                keypoints_id.append(I)

        elif D4_min[I] == np.min(collected_subset_min) or D4_min[I] == np.max(collected_subset_min):
            if check_keypoint(p, collected_subset_min, collected_subset_max, contrast_threshold, bounds, ref_mesh):
                keypoints_id.append(I)

        elif D2_max[I] == np.min(collected_subset_max) or D2_max[I] == np.max(collected_subset_max):
            if check_keypoint(p, collected_subset_min, collected_subset_max, contrast_threshold, bounds, ref_mesh):
                keypoints_id.append(I)

        elif D3_max[I] == np.min(collected_subset_max) or D3_max[I] == np.max(collected_subset_max):
            if check_keypoint(p, collected_subset_min, collected_subset_max, contrast_threshold, bounds, ref_mesh):
                keypoints_id.append(I)

        elif D4_max[I] == np.min(collected_subset_max) or D4_max[I] == np.max(collected_subset_max):
            if check_keypoint(p, collected_subset_min, collected_subset_max, contrast_threshold, bounds, ref_mesh):
                keypoints_id.append(I)
                        
                        
    #Remove redundancy
    keypoints_id =list(set(keypoints_id))

    # Collect possible keypoints
    keypoints.append(submesh.points()[keypoints_id])
    #print("    Amount of key points: ",len(keypoints_id)," ~ ",round((len(keypoints_id)/len(mesh.points()))*100,2)," %              ")

    # check principle curvature maps
    if inspection == True:
        visual_mesh = submesh.clone()
        centers = submesh.points()[keypoints_id]
        sp = vedo.Spheres(centers=centers, r = 0.5)
        visual = vedo.merge(sp,visual_mesh)
        submesh.mapPointsToCells()
        vedo.write(submesh,os.path.join(output, f"TEST_{name}_.vtp"))
            
    #Reformat to non-nested list
    keypoints = keypoints[0]
    
    # Check keypoints on mesh
    if inspection == True:
        visual_mesh = submesh.clone()
        centers = keypoints
        sp = vedo.Spheres(centers=centers, r = 0.5)
        visual = vedo.merge(sp,visual_mesh)
        vedo.write(visual,os.path.join(output, f"keypoint_detect_final_{name}.stl"))
        
     
    #Initialize output 
    outputPts = []
    outputIdx = []
    
    #Get keypoint-IDXs from original mesh
    if returnIdx == True:
        for kp in keypoints:
            outputIdx.append(mesh.closestPoint(kp,returnPointId=True))
    #Get keypoint coordinates from original mesh
    if returnPts == True:
        for kp in keypoints:
            outputPts.append(mesh.closestPoint(kp))
    #Get keypoint coordinates if nothing is specified
    if returnIdx == False and returnPts == False:
        #print("Cannot return nothing. Points will be returned")
        for kp in keypoints:
            outputPts.append(mesh.closestPoint(kp))

    if returnIdx == True and returnPts == True:
        return outputIdx, outputPts
    elif returnIdx == True and returnPts == False:
        return outputIdx
    elif returnIdx == False and returnPts == True:
        return outputPts    
    elif returnIdx == False and returnPts == False:
        return outputPts

    