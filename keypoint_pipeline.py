###################################################
###            Keypoint pipeline                ###
###################################################
# Author: Anika Kofod Petersen
# Date: 3rd of March 2025
# Tested using python 3.10.6

# Import packages
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scipy') 
import os
import sys
import vedo
import time
import json
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
from scipy.spatial import KDTree
import pyshot
from scipy.spatial import procrustes
import json


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
    distances = np.linalg.norm(mesh_points-point,axis=1)
    keep_index = np.where(distances < r)[0]
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
    

def check_keypoint(p, collected_subset_min, collected_subset_max, contrast_threshold, bounds, ref_mesh):
    # Precompute the contrast difference
    contrast_diff = np.max(collected_subset_min) - np.min(collected_subset_min)
    
    # Check if contrast is above threshold
    if contrast_diff > contrast_threshold:
        # Get the coordinates of boundary points at once
        boundary_points = np.array([ref_mesh.points()[bound_id] for bound_id in bounds])
        
        # Compute the distances between p and all boundary points
        distances = np.linalg.norm(boundary_points - p, axis=1)
        
        # Check if any distance is less than the threshold (distance threshold = 1)
        if np.any(distances < 1):
            return False
        
        # No boundary point is too close, return True
        return True
        
    return False

    
def compute_DoG_meshes(ref_mesh, sigmas=[0.1, 0.2, 0.3, 0.5, 0.8, 1.0], radius=4, batch_size=1000):
    # Initialize the list for the DoG meshes
    dog_meshes = [ref_mesh.clone() for _ in range(len(sigmas))]
    total = len(ref_mesh.points())
    
    # Build the k-d tree for fast nearest neighbor queries
    points = ref_mesh.points()
    kdtree = KDTree(points)
    
    # Process the mesh in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_points = points[batch_start:batch_end]
        
        print(f"Processing points {batch_start} to {batch_end} ({round((batch_end / total) * 100, 2)}%)", end="\r")
        
        # For each point in the current batch
        for i, p in enumerate(batch_points):
            idx_neigh = kdtree.query_ball_point(p, radius)  # Find all points within the radius
            distances = np.linalg.norm(points[idx_neigh] - p, axis=1)

            # Calculate the Gaussian weights for each sigma
            weights = np.array([gaussian(distances, 0, sigma) for sigma in sigmas])

            # Calculate smoothed points for each sigma
            smoothed_points = [np.average(points[idx_neigh], weights=w, axis=0) for w in weights]

            # Update the DoG meshes for each sigma
            for j, dog_mesh in enumerate(dog_meshes):
                dog_mesh.points()[batch_start + i] = smoothed_points[j]


    return dog_meshes

def gaussian(x, mean, sigma):
    """Gaussian function to calculate weights."""
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

def fast_DoG_smoothing(ref_mesh, sigmas, radius, dog_meshes):
    points = ref_mesh.points()
    total = len(points)

    # Build KDTree once
    kdtree = KDTree(points)

    # Iterate through points using batch processing for efficiency
    for i, p in enumerate(points):
        if i % 100 == 0:  # Reduce console prints for speedup
            print(f"Processing: {round((i + 1) / total * 100, 2)}%   ", end="\r")

        # Get neighbors within radius
        connect_idx = kdtree.query_ball_point(p, radius)

        # Extract neighbor points
        neighbor_points = points[connect_idx]

        # Compute distances in one batch operation
        distances = np.linalg.norm(neighbor_points - p, axis=1)

        # Compute weights for all sigmas
        weights = np.exp(-((distances[:, None]) ** 2) / (2 * np.array(sigmas) ** 2))

        # Normalize weights
        weights /= weights.sum(axis=0, keepdims=True)

        # Compute smoothed points using weighted averages
        smoothed_points = np.dot(weights.T, neighbor_points)

        # Update dog_mesh points
        for j, dog_mesh in enumerate(dog_meshes):
            dog_mesh.points()[i] = smoothed_points[j]

    return dog_meshes
    
def keypoint_detection(mesh, name = "test", res=20, radius = 2, returnIdx = False, returnPts = True, inspection = True, output="./"):
    """ Function for detecting robust keypoints"""
    """
    Input: mesh to find keypoints on, a name for output files, output path, 
    Output: keypoints
    """
    
    # Collection of possible keypoints, initialization
    keypoints = []
    
    # Decimate for reproducibility
    print("Decimating/subdividing")
    teeth = mesh.clone()
    area = teeth.area()
    n = len(teeth.points())
    n_a = round(res*area)
    frac = n_a/n
    print("Aimed resolution: ",round(n_a/area,3))
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
    print("True resolution: ",round(len(teeth.points())/area,3))
    
    # Work with a mesh clone
    submesh = teeth.clone()

    # Apply Gaussian smoothing to deal with Difference of Curvature (DoC) - Initialization
    print("Preparing DoGs")
    DoG = []
    dog_meshes = [submesh.clone() for _ in range(6)]
    ref_mesh = submesh.clone()
    total = len(ref_mesh.points())
    sigmas = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

    dog_meshes = fast_DoG_smoothing(ref_mesh, sigmas, radius, dog_meshes)
    """
    # Iterate through points in mesh
    for i, p in enumerate(ref_mesh.points()):
        print("    ", round(((i + 1) / total) * 100, 2), " %   ", end="\r")

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
    """
    # Collect the smoothed meshes
    DoG.extend(dog_meshes)
    if inspection == True:
        for i in range(len(sigmas)):
            vedo.write(DoG[i],os.path.join(output, f"smoothmesh_{name}_{i}_.stl"))

                
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
    print("    Working with DoG pyramids                                     ")
    # Precompute the min/max values for D1, D2, ..., D5
    D_min = [np.subtract(dog_min[i], dog_min[i + 1]) for i in range(5)]
    D_max = [np.subtract(dog_max[i], dog_max[i + 1]) for i in range(5)]

    # Initialize keypoint detection
    bounds = boundaries(ref_mesh, return_point_ids=True)
    keypoints_id = []
    contrast_threshold = 0.00001

    # Iterate through points in mesh
    for I, p in enumerate(ref_mesh.points()):
        print("    ", round(((I + 1) / total) * 100, 2), " %   ", end="\r")
    
        # Compare closest neighborhood
        main_idx, connect_idx = find_subset(I, ref_mesh, ring=1)
        connect_idx = list(set(sum(connect_idx, [])))

        # Collect min/max subsets
        collected_subset_min = np.concatenate([D_min[i][connect_idx] for i in range(3)], axis=0)
        collected_subset_max = np.concatenate([D_max[i][connect_idx] for i in range(3)], axis=0)

        min_collected_min = np.min(collected_subset_min)
        max_collected_min = np.max(collected_subset_min)
        min_collected_max = np.min(collected_subset_max)
        max_collected_max = np.max(collected_subset_max)

        # Check keypoints only for relevant conditions
        if D_min[1][I] == min_collected_min or D_min[1][I] == max_collected_min or \
           D_min[2][I] == min_collected_min or D_min[2][I] == max_collected_min or \
           D_min[3][I] == min_collected_min or D_min[3][I] == max_collected_min or \
           D_max[1][I] == min_collected_max or D_max[1][I] == max_collected_max or \
           D_max[2][I] == min_collected_max or D_max[2][I] == max_collected_max or \
           D_max[3][I] == min_collected_max or D_max[3][I] == max_collected_max:

            # Check if this point is a keypoint
            if check_keypoint(p, collected_subset_min, collected_subset_max, contrast_threshold, bounds, ref_mesh):
                keypoints_id.append(I)

    # Remove redundancy
    keypoints_id = list(set(keypoints_id))

    # Collect possible keypoints
    keypoints.append(submesh.points()[keypoints_id])
    print("    Amount of key points: ",len(keypoints_id)," ~ ",round((len(keypoints_id)/len(mesh.points()))*100,2)," %              ")

    # check principle curvature maps
    if inspection == True:
        visual_mesh = submesh.clone()
        centers = submesh.points()[keypoints_id]
        sp = vedo.Spheres(centers=centers, r = 0.5)
        visual = vedo.merge(sp,visual_mesh)
        submesh.mapPointsToCells()
        vedo.write(submesh,os.path.join(output, f"TEST_{name}_.stl"))
            
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


def calculate_SHOT(keypoints,mesh, radius=1):
    """ Function for calculating SHOT descriptor
    NOTICE: the keypoints should be given as point idx
    Also notice the usage of the pyshot package
    """
    """
    Calculates the  Signature of Histograms of OrienTations descriptor given keypoints and mesh surface.
    
    Parameters:
        mesh (vedo mesh): A mesh object representing the vertices and faces of the mesh surface.
        keypoints (numpy array):A list of point indexes representing the keypoints of interest.
        radius (float): The radius within which to consider points for the SHOT calculations.
    
    Returns:
        shot_hist (numpy array): An dictionary of arrays representing the sfh histograms.
    """
    #print("Calculating Shot descriptors                                  ")
    vertices: np.array =  np.array(mesh.points().astype(np.float64))
    faces: np.array =  np.array(mesh.faces())

    # a np.array of shape (n, n_descr)
    shot_descrs: np.array = pyshot.get_descriptors(
        vertices,
        faces,
        radius=radius,
        local_rf_radius=radius,
        # The following parameters are optional
        min_neighbors=3,
        n_bins=10,
        double_volumes_sectors=True,
        use_interpolation=True,
        use_normalization=True,
        )
    keypoint_descrs = shot_descrs[keypoints]
    #keypoint_descrs = np.array([list(map(normalize,keypoint_descrs))])  #normalize
    keypoint_descrs = dict(enumerate(keypoint_descrs))
    keypoint_descrs = {dict_key:list(v) for dict_key, (k, v) in enumerate(keypoint_descrs.items())}

    return keypoint_descrs

def save_keypoints(keypoints, name, output_path):
    with open(os.path.join(output_path,name+".json"), "w") as json_file:
        json.dump(keypoints, json_file)

def convert_to_array(dictionary):
    '''Converts lists of values in a dictionary to numpy arrays'''
    return {k:np.array(v) for k, v in dictionary.items()}

def keypoint_correspondence(dict1, dict2, coord1, coord2):
    ''' Function for finding landmark correspondence'''
    ''''
    Input: two landmark dicts with flat matrices as values
    Output: landmark key correspondence'''
    #convert
    dict1 = convert_to_array(dict1)
    dict2 = convert_to_array(dict2)

    #Initialize
    distance_ratios = []
    best_keys_target = []
    best_keys_query = []

    #Iterate through first dict
    for i, (key1, value1) in enumerate(dict1.items()):
        #Initialize
        dist_list = []

        #Get distance to all in second dict
        dist_list = np.linalg.norm((list(dict2.values())-value1),axis=1)

        #Sort landmarks according to similarity
        dists_sorted, keys_matches = zip(*sorted(zip(dist_list, dict2.keys())))

        #Ratio between best match and second best match
        if dists_sorted[1] != 0 and dists_sorted[1] != dists_sorted[0]:
            L2_ratio = dists_sorted[0]/dists_sorted[1]
            distance_ratios.append(L2_ratio)
        else:
            L2_ratio = 1
            distance_ratios.append(L2_ratio)

        # Collect the best match query key (key1) and the best match target key (key from sfh_dict2)
        best_keys_query.append(key1)  # Best key from the query dictionary
        best_keys_target.append(keys_matches[0])  # Best key from the target dictionary


    # Sort
    dist_sorted, keys_query_sorted, keys_target_sorted = zip(*sorted(zip(distance_ratios, best_keys_query, best_keys_target)))
    dist_sorted = list(dist_sorted)
    keys_query_sorted = list(keys_query_sorted)
    keys_target_sorted = list(keys_target_sorted)
    
    # iterate through ranges
    ar_mean = []
    disparity = []

    # Get coordinates
    query_coords = coord1
    target_coords = coord2

    # create subsets
    rr = 15
    rr_subset = dist_sorted[:rr]
    query_kp = keys_query_sorted[:rr]
    target_kp = keys_target_sorted[:rr]

    # Means
    ar_mean.append(np.mean(rr_subset))

    # disparity subset
    rr = 8
    query_kp = keys_query_sorted[:rr]
    target_kp = keys_target_sorted[:rr]

    # Disparity
    query_dis = [query_coords[key] for key in query_kp]
    target_dis = [target_coords[key] for key in target_kp]
    if all(np.array_equal(element, target_dis[0]) for element in target_dis):
        disparity.append(1)
    else:
        _, _, disp_raw = procrustes(np.array(query_dis), np.array(target_dis))
        disparity.append(disp_raw)

    score = 0.9*ar_mean[0]+0.1*disparity[0]

    return score
