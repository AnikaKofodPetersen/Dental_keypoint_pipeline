#################################
###    Grid Cutting Method    ###
###    Method for removing    ###
### soft tissue from 3D scans ###
#################################
# Author: Anika Kofod Petersen
# Date: 6th November 2023
# Extraordinary Dependencies: vedo, vtk, sklearn, numpy, scipy
# Tested using python 3.10.6


#Import packages
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scipy') 
import os
import sys
import vedo
import argparse
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util.numpy_support import vtk_to_numpy 
import numpy as np
import scipy
from sklearn.decomposition import PCA
import math
import time
from collections import Counter


#Define orienting function
def orient_mesh(full_mesh, teeth_type = "man"):
    """ A function to orienting the mesh properly according to X,Y and Z plane"""
    """
    Input: full mesh, orientation of teeth compared to z-axis (mandibular/maxilla),
    Output: full mesh (oriented)
    """
    
    # Change directions
    if teeth_type.lower()=="max" or teeth_type.lower() == "maxilla":
        print("Changing orientation.")
        full_mesh = full_mesh.rotate(180,axis=(0,1,0))  
    elif teeth_type.lower()=="man" or teeth_type.lower() == "mandibule":
        print("No orientation changes needed.")
    else:
        print("Did not understand teeth type. No orientation changes performed.")
 
    return full_mesh   


#Function for finding occlusal points
def occlusalplane_points(teeth_mesh, inspection = False, output = "./",name="test"):
    """ Sectionize teeth and find starting points for suiting occlusal plane"""
    """
    Input: decimated mesh of teeth
           Wether or not the results should be manually inspected with plots,
           output path and name
    Output: 3 most extreme points from the same mesh cut into 4 sections
            according to a fit plane, a symmetry plane and
            the orthorgonal symmetry plane
    """
    print("Finding occlusal plane")
    
    #Make clones
    cut_mesh = teeth_mesh.clone()
    fig_mesh = teeth_mesh.clone()
    
    # Make initial fit plane
    org_points = teeth_mesh.points().copy()
    fitplane = vedo.fitPlane(org_points)
    
    #Check fitplane orientation
    if fitplane.normal[2] < 0:
        fitplane = vedo.Plane(pos=fitplane.centerOfMass(), normal= -1*fitplane.normal, sx=100, sy = 100)
    
    # Define symmetry plane and othogonal symmetry plane (X and Y plane)
    symplane = vedo.Plane(pos=teeth_mesh.centerOfMass(), normal=(1,0,0), sx=100, sy=100)
    ort_symplane = vedo.Plane(pos=fitplane.centerOfMass(), normal=(0,1,0), sx=100, sy=100)
    sym_n = (1,0,0)
    orth_n = (0,1,0)
    
    # Cut off bottom
    u_bottom = cut_mesh.clone()
    u_bottom = u_bottom.cutWithPlane(origin=fitplane.centerOfMass(), normal=fitplane.normal)
    
    # Figure out where to do trisections
    center = np.array(((max(u_bottom.points()[:,0])-abs(min(u_bottom.points()[:,0])))/2.5,(max(u_bottom.points()[:,1])-abs(min(u_bottom.points()[:,1])))/2,(max(u_bottom.points()[:,2])-abs(min(u_bottom.points()[:,2]))/2)))    
    tri_rangeA = ((max(u_bottom.points()[:,0])-min(u_bottom.points()[:,0]))/4)/2
    tri_rangeB = ((max(u_bottom.points()[:,1])-min(u_bottom.points()[:,1]))/4)/2
    

    #Disect into trisections (6 in total)
    #First, cut into sections on x-axis
    tri1 = u_bottom.clone()
    tri1 = tri1.cutWithPlane(origin=(center+np.array((tri_rangeA,0,0))), normal=symplane.normals()[0])
    tri2 = u_bottom.clone()
    tri2 = tri2.cutWithPlane(origin=(center+np.array((tri_rangeA,0,0))), normal=-symplane.normals()[0])
    tri3 = tri2.clone()
    tri2 = tri2.cutWithPlane(origin=(center-np.array((tri_rangeA,0,0))), normal=-symplane.normals()[0])
    tri3 = tri3.cutWithPlane(origin=(center-np.array((tri_rangeA,0,0))), normal=symplane.normals()[0])
    
    #The cut into sections on y-axis (Section 1)
    tri1A = tri1.clone()
    tri1A = tri1A.cutWithPlane(origin=(center+np.array((0,-tri_rangeB,0))), normal=ort_symplane.normals()[0])
    tri1B = tri1.clone()
    tri1B = tri1B.cutWithPlane(origin=(center+np.array((0,-tri_rangeB,0))), normal=-ort_symplane.normals()[0])
    
    #The cut into sections on y-axis (Section 2)
    tri2A = tri2.clone()
    tri2A = tri2A.cutWithPlane(origin=(center+np.array((0,-tri_rangeB,0))), normal=ort_symplane.normals()[0])
    tri2B = tri2.clone()
    tri2B = tri2B.cutWithPlane(origin=(center+np.array((0,-tri_rangeB,0))), normal=-ort_symplane.normals()[0])
    
    #The cut into sections on y-axis (Section 3)
    tri3A = tri3.clone()
    tri3A = tri3A.cutWithPlane(origin=(center+np.array((0,-tri_rangeB,0))), normal=ort_symplane.normals()[0])
    tri3B = tri3.clone()
    tri3B = tri3B.cutWithPlane(origin=(center+np.array((0,-tri_rangeB,0))), normal=-ort_symplane.normals()[0])
    
    # Make proper planes for visuals
    tri_plane1A = vedo.Plane(pos=(center+np.array((tri_rangeA,0,0))), normal=-symplane.normals()[0], sx=100, sy=100)
    tri_plane1B = vedo.Plane(pos=(center-np.array((tri_rangeA,0,0))), normal=-symplane.normals()[0], sx=100, sy=100)
    tri_plane2A = vedo.Plane(pos=(center+np.array((0,-tri_rangeB,0))), normal=-ort_symplane.normals()[0], sx=100, sy=100)
        
    # Find points with largest distance to fitplane (one for each section)
    dist = []
    sec_points = []
    for sec in [tri1A,tri1B,tri2A,tri2B,tri3A,tri3B]: 
        all_dist = [shortest_distance(p,fitplane,fitplane.centerOfMass()) for p in sec.points()]
        if len(all_dist) > 1:
            max_dist = np.argmax(all_dist)
            sec_points.append(sec.points()[max_dist])
            dist.append(all_dist[max_dist])
        
    #Sort points accoring to distances (z-coordinate)
    sort_dist, sort_point = zip(*sorted(zip(dist, sec_points),reverse=True))
    sort_point = sorted(sec_points, key=lambda x: x[2], reverse=True)
    chosen_points = sort_point[:4]

    # Inspection plot if needed (spheres marking chosen points)
    if inspection == True:
        
        #Chosen points
        sph = vedo.Spheres(centers= sort_point, r=1)
        sph_chosen = vedo.Spheres(centers= chosen_points, r=2)
        inspect = vedo.merge(sph, teeth_mesh,sph_chosen)
        vedo.write(inspect, os.path.join(output, f'{name}_inspect_chosen_points.stl'))
        
        #Fitplane
        section_planes = vedo.merge(fig_mesh, fitplane, sph)
        vedo.write(section_planes,os.path.join(output, f'{name}_inspect_fitplane.stl') )
        
        #Trisections
        tri_sections = vedo.merge(tri_plane1A, tri_plane1B, tri_plane2A,  sph, fig_mesh, fitplane) 
        vedo.write(tri_sections,os.path.join(output, f'{name}_inspect_trisections.stl') )

        # Seperate sections    
        for i,sec in enumerate([tri1A,tri1B,tri2A,tri2B,tri3A,tri3B]): 
            sec_out = vedo.merge(sph, sec, fitplane)
            vedo.write(sec_out, os.path.join(output, f'{name}_inspect_trisections{i}.stl'))
        
    # return points and sections
    return chosen_points

def square_filter(mesh_points,extremes):
    """ Filter for ignoring all points out of x/y range,
        thus cutting out a square"""
    """
    Input: mesh points and list of [xmin, xmax, ymin, ymax] 
    Output: indexes of points within square
    """

    # Get list of point-coordinate in x/y direction
    mesh_points = np.array(mesh_points)
    x_coord = mesh_points[:,0]
    y_coord = mesh_points[:,1]
    
    # Get indexes within square limit for each direction (x/y)
    idx_x = np.where((x_coord<extremes[1]) & (x_coord>extremes[0]))[0]
    idx_y = np.where((y_coord<extremes[3]) & (y_coord>extremes[2]))[0]
    
    # Get intersect (square)
    intersect = idx_x[np.in1d(idx_x, idx_y)]
    return intersect


#Define function for shortest distance
def shortest_distance(point, plane, plane_point):
    """Find shortest distance from point to plane."""
    """ 
    Input: 3D point and plane
    Output: distance from point to plane
    """

    a, b, c = plane.normals()[0]
    x0, y0, z0 = plane_point
    d = -(a*x0 + b*y0 + c*z0)
    x1, y1, z1 = point
    d1 = abs((a * x1 + b * y1 + c * z1 + d))
    e = (math.sqrt(a * a + b * b + c * c))
    
    return d1/e


def analyse_square(square_idx, square_pts, occlusal, norm, dist_thres = 10, inclusion_thres = 5):
    """ Analyse what points to remove from specific square"""
    """
    Input: indexes of points within square, points within square, 
           occlusal plane, normal vector to occlusal plane in negative z direction,
           distance threshold from occlusal plane to the top point of the square,
           inclusion threshold defining how much of the square is included from the top point and down
    Output: list of indexes to be deleted
    """
    # Collect z-coordinates
    z_coord = square_pts[:,2]
    
    # Find top point and the distance to the occlusal plane
    top_point = square_pts[np.argmax(z_coord)]
    dist = shortest_distance(top_point, occlusal, occlusal.centerOfMass())
    
    # If distance within reach of occlusal plane
    if dist < dist_thres:
        
        # move top point in the direction of the normal vector 
        deletion_plane = top_point+inclusion_thres*norm
        
        # Collect indexes that are further away than the threshold
        idx_z = np.where((z_coord<deletion_plane[2]))[0]
        del_id = square_idx[idx_z]
        
        
    # Collect all indexes if top is too far away from occlusal plane
    else:
        del_id = square_idx
        
    #return deletion indexes
    return del_id
    
    
#Function for calculating connectivity
def compute_connectivity(self, radius = 0.25):
    """
    Flag a mesh by connectivity: each disconnected region will receive a different Id.
    You can access the array of ids through ``mesh.pointdata["RegionId"]``.
    Adaptation form vedo package
    """
    cf = vtk.vtkConnectivityFilter()
    cf.SetInputData(self.polydata(False))
    cf.SetExtractionModeToAllRegions()
    cf.ColorRegionsOn()
    cf.Update()
    return self._update(cf.GetOutput())


def Remove_islands(mesh,p=0.1): #0.07
    """ Function for removing small islands"""
    """
    Input: mesh, size threshold for being considered an island (default 10%)
    Output: mesh with no small islands
    """
    
    #Get region ids for connected islands
    Connected_mesh = compute_connectivity(mesh)
    RegionID = dict(Counter(Connected_mesh.celldata["RegionId"]))
    
    # Find largest island
    maxi = max(RegionID.values())
    dele = []
    
    # Flag all islands less than 10% of the size of the biggets island
    for idx,count in RegionID.items():
        if count < maxi*p:
            dele.append(idx)
    
    # Delete flagged islands
    del_cells = [i for i,x in enumerate(Connected_mesh.celldata["RegionId"]) if x in dele]
    clean_mesh = mesh.clone().deleteCells(del_cells)
    clean_mesh.clean()

    return clean_mesh


#Define decimating function
def decimate_mesh(full_mesh,target_num=5000):
    """ A function for performing quadratic decimation """
    """
    Input: full mesh and target number of faces
    Output: decimated mesh
    """
    #Decimate
    ratio = target_num/full_mesh.NCells() # calculate ratio
    mesh_d = full_mesh.clone()
    mesh_d.decimate(fraction=ratio)
    
    #Sanity check - redo if specific number of faces went wrong
    if len(full_mesh.faces())*(ratio+0.2) < len(mesh_d.faces()):
        ratio = (target_num+100)/full_mesh.NCells()
        mesh_d = full_mesh.clone()
        mesh_d.decimate(fraction=ratio)
    return mesh_d

# Final main function for cutting off soft tissue
def cut_mesh(o_teeth, ratio = 0.33, cut_resolution= 3, inclusion_thres = 5, re_orient = True, name ="test",out_path="./", inspection = False, teeth_type = "man"):
    """ Collective function for cutting mesh
    Input: oriented mesh, size of squares, distance threshold and inclusion threshold
    Output: cut mesh
    """
    
    # Find occlusal plane
    o_teeth_d1 = decimate_mesh(o_teeth)
    unique_points = occlusalplane_points(o_teeth_d1,inspection = inspection, output=out_path, name=name)
    occlusal = vedo.fitPlane(np.array(unique_points))
    
    # Return results for visuals
    if inspection == True:
        sph = vedo.Spheres(centers=unique_points, r=0.5)
        temp_mesh = o_teeth.clone()
        mm = vedo.merge(temp_mesh, occlusal,sph)
        vedo.write(mm,os.path.join(out_path, f"{name}_occlusal.vtp"))
        vedo.write(o_teeth_d1,os.path.join(out_path, f"{name}_decimated.vtp"))
    
    # Determine direction of normal vector
    norm = occlusal.normal
    if norm[2] > 0:
        norm = -1*norm

    # Find bounding box for dentition
    bounds = o_teeth.bounds()

    # Collect range of mesh in euclidean directions
    x_range = np.arange(bounds[0]-cut_resolution,bounds[1]+cut_resolution,cut_resolution)
    y_range = np.arange(bounds[2]-cut_resolution,bounds[3]+cut_resolution,cut_resolution)
    z_range = bounds[5]-bounds[4]
    dist_thres = ratio*z_range
    
    # Initialize empty array
    del_ids = np.array([])

    # Iterate through grid squares
    for i,x_coord in enumerate(x_range[:-1]):
        for j,y_coord in enumerate(y_range[:-1]):
        
            # Collect ranges for grid squares
            x_int = [x_range[i],x_range[i+1]]
            y_int = [y_range[j],y_range[j+1]]
            
            # Collect grid squares
            square_idx = square_filter(o_teeth.points(),x_int+y_int)
            square_pts = o_teeth.points()[square_idx]
            
            # Only work with non-empty squares
            if len(square_idx) > 0:
                
                #Collect point indexes to be deleted
                del_id = analyse_square(square_idx, square_pts, occlusal, norm, dist_thres = dist_thres, inclusion_thres = inclusion_thres)
                del_ids = np.append(del_ids, del_id)
            
    # Deleting appropriate points
    print("Deleting ",len(del_ids),"/", len(o_teeth.points()), 
          " points (",round((len(del_ids)/len(o_teeth.points()))*100,2),"% )")
    del_ids = [*map(int, del_ids)]
    o_teeth = o_teeth.deletePoints(indices=del_ids)
    o_teeth.clean()
    
    #Delete floating islands
    o_teeth = Remove_islands(o_teeth)
    
    # Reorient mesh
    if re_orient == True:
        o_teeth = orient_mesh(o_teeth, teeth_type = teeth_type)

    # Return results
    vedo.write(o_teeth,os.path.join(out_path, f"{name}.stl"))
        
    return o_teeth


# The actual script for running
if __name__ == '__main__':
    
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-A', '--arch_type',
                    default='man',
                    choices=('man', 'max'),
                    help="Arch type: Either 'man' for mandible or 'max' for maxilla. Default: man ",
                    )
    parser.add_argument('-I', '--inclusion',
                    default=8,
                    help="I value for inclusion threshold. Default: 8 ",
                    type=float
                    )
    parser.add_argument('-G', '--grid_size',
                    default=5,
                    help="Grid size. Default: 5 ",
                    type=float
                    )
    parser.add_argument('-R', '--ratio_of_inclusion',
                    default=0.33,
                    help="Ratio of z range for auto-inclusion",
                    type=float
                    )
    parser.add_argument('--inspection',
                    action='store_true',
                    help='Save intermediate files for inspection.'
                    )
    parser.add_argument('-IN', '--input_path',
                    default="./",
                    help="Path to input file.",
                    type=str
                    )
    parser.add_argument('-OUT', '--output_path',
                    default="./",
                    help="Path to output directory.",
                    type=str
                    )
    parser.add_argument('-N', '--name',
                    default="grid_cutting",
                    help="Name of the final dentition model after cutting.",
                    type=str
                    )
    args = parser.parse_args()
    inspection = args.inspection

    
    # Load, orient and cut dental mesh
    teeth = vedo.io.load(args.input_path)
    o_teeth = orient_mesh(teeth, teeth_type = args.arch_type)
    cutmesh = cut_mesh(o_teeth, ratio = args.ratio_of_inclusion, cut_resolution= args.grid_size, inclusion_thres = args.inclusion, name=args.name,out_path=args.output_path)
    
    print(f"The final dentition model has been saved at {args.output_path}/{args.name}.stl")

    
    
