import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import operator
from PIL import Image
from plyfile import PlyData, PlyElement

def read_data(location):
    """
    Read .csv or .ply data into a 2d array

    Parameter
    ---------
    location : string
        path where the point data is stored

    Returns
    -------
    coord_data : 2d array
        coordniates of all points e.g., [[x1 = 1.3, y1 = 2.54, z1 = 2.31][...]]
    """
    datatype =location[-3] + location[-2] + location[-1]
    if datatype == "csv":
        with open(location, encoding='UTF8') as f:         #reading coordinates from csv document
            csv_reader = csv.reader(f)
            next(csv_reader)                                        #skips first row (x,y,z)
            coord_data = []
            coord_pre= []
            for line in csv_reader:
                coord_pre.append(float(line[0]))                    #coordinates combined
                coord_pre.append(float(line[1]))
                coord_pre.append(float(line[2]))
                coord_data.append(coord_pre)
                coord_pre = []
    
    elif datatype == "ply":
        with open(location, 'rb') as f:
            plydata = PlyData.read(f)

        ply_coord_data = []
        ply_coord_data.append(plydata.elements[0].data["x"])
        ply_coord_data.append(plydata.elements[0].data["y"])
        ply_coord_data.append(plydata.elements[0].data["z"])

        ply_coord_data_format = []
        coord = []
        j = 0
        while j < len(ply_coord_data[0]):
            coord.append(np.round(float(ply_coord_data[0][j])))                    #coordinates combined
            coord.append(np.round(float(ply_coord_data[1][j])))
            coord.append(np.round(float(ply_coord_data[2][j])))
            ply_coord_data_format.append(coord)
            coord = []
            j += 1
        coord_data = ply_coord_data_format
    else:
        sys.exit("Data type has to be .csv or .ply")

    return coord_data

def reduce_data(coord_data):
    """
    Reduces data to around 20 million points

    Parameter
    ---------
    coord_data : 2d array
        coordniates of all points e.g., [[x1 = 1.3, y1 = 2.54, z1 = 2.31][...]]

    Returns
    -------
    coord_data_reduced : 2d array
        coordniates of points after reduction
    """
    coord = []
    coord_data_reduced = []
    i = 0
    j = 0
    while j < len(coord_data):
        if i >= len(coord_data) / 20000000:
            coord.append(float(coord_data[j][0]))                    #coordinates combined
            coord.append(float(coord_data[j][1]))
            coord.append(float(coord_data[j][2]))
            coord_data_reduced.append(coord)
            coord = []
            i = 0
        else:
            i += 1
        j += 1
    return coord_data_reduced

def probe_top_edge(filename_image):
    """
    Return position of top and lower edge of the probe. Also returns array of picture

    Parameter
    ---------
    filename_image : string
        path where the image file is stored

    Returns
    -------
    modified_array : 2d array
        array of colur values for picture.
        Can by printed with
            im = Image.fromarray(modified_array)
            im.show
        Edges are marked
    top_mean_mean : int
        vertical distance between top edge of picture and average of top edge of the probe
    lower_mean_mean : int
        vertical distance between top edge of picture and average of lower edge of the probe
    """
    z = 0
    lower_mean_sum = 0
    top_mean_sum = 0
    while z < 10:


        im = Image.open(filename_image[z])
        imarray = np.array(im)
        imarray_T = np.transpose(np.array(im))

        i = 0
        start_left = len(imarray[0])
        while i < len(imarray):
            j = 0
            while j < len(imarray[i]):
                if imarray[i][j] == 255 and j < start_left:
                    start_left = j
                    break
                j += 1
            i += 1

        i = 0
        end_right = 0
        while i < len(imarray):
            j = 0
            while j < len(imarray[i]):
                if imarray[i][j] == 255 and j > end_right:
                    end_right = j
                j += 1
            i += 1  

        width_distance = end_right - start_left
        imarray_T[imarray_T == 255] = 50

        top = []
        i = int(start_left + 0.1 * width_distance)
        while i < len(imarray_T):
            j = 0
            while j < len(imarray_T[i]):
                if imarray_T[i][j] == 50:
                    imarray_T[i][j] = 254
                    j += 1
                    top.append(j)
                    break
                else:
                    j += 1
            if i == int(end_right - 0.1 * width_distance):
                break
            i += 1

        top_mean = int(np.mean(top))

        lower = []
        i = int(start_left + 0.1 * width_distance)
        while i < len(imarray_T):
            j = len(imarray_T[i]) - 1
            while j > 0:
                if imarray_T[i][j] == 50:
                    imarray_T[i][j] = 254
                    j -= 1
                    lower.append(j)
                    break
                else:
                    j -= 1
            if i == int(end_right - 0.1 * width_distance):
                break
            i += 1

        lower_mean = int(np.mean(lower))

        k = 0
        while k < len(imarray_T):
            h = 0
            while h < len(imarray_T[k]):
                if h == top_mean or h == lower_mean:
                    imarray_T[k][h] = 255
                h += 1
            k += 1
        imarray_TT = np.transpose(imarray_T)

        lower_mean_sum += lower_mean
        top_mean_sum += top_mean
        print("lower:", lower_mean_sum)
        print("top:", top_mean_sum)

        if z == 0:
            modified_array = np.copy(imarray_TT)
            modified_array = np.stack((modified_array, np.zeros_like(modified_array), np.zeros_like(modified_array)), axis=-1)
            modified_array[modified_array[..., 0] == 255] = [204,7,30]  #red
            modified_array[modified_array[..., 0] == 254] = [0,84,159]  #blue
            modified_array[modified_array[..., 0] == 50] = [100, 100, 100]   #grey
        z += 1
    top_mean_mean = top_mean_sum / z
    lower_mean_mean = lower_mean_sum / z

    return modified_array, top_mean_mean, lower_mean_mean

def fibre_top_edge(filename_image):
    """
    Return distance between top edge of picture and top edge of the fibre

    Parameter
    ---------
    filename_image : string
        path where the image file is stored

    Returns
    -------
    top : int
        vertical distance between top edge of picture and first white pixel
    """
    im = Image.open(filename_image)
    imarray = np.array(im)
    imarray_T = np.transpose(np.array(im))

    i = 0
    top = len(imarray_T[0])
    while i < len(imarray_T):
        j = 0
        while j < len(imarray_T[i]):
            if imarray_T[i][j] == 255 and j < top:
                top = j
                j += 1
                break
            else:
                j += 1
        i += 1

    return top

def get_medium(coordinates):
    """
    Return medium of givin values (like np.mean?)

    Parameter
    ---------
    coordinates : 1d array
        array of float values

    Returns
    -------
    medium : float
        medium of given values
    """

    j = 0
    temp = 0
    z_gesamt = 0
    while j < len(coordinates):
        temp += coordinates[j]
        j += 1
    medium = temp / len(coordinates)
    return medium

def max_point (axis):
    "no londer in use"

    if axis == "x":
        temp = 0
    elif axis == "y":
        temp = 1
    elif axis == "z":
        temp = 2
    with open('coordinates.csv', encoding='UTF8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader);               #skips first row (x,y,z)
        max = 0        
        for line in csv_reader:
            if float(line[temp]) > max:
                max = float(line[temp])
            else:
                next 
    return max

def get_edge_controlpoints_old(x_values, y_values, coord_data):
    "no londer in use"


    #print("\n Nächster Durchlauf", x_values)
    if any(x_values) is False:
        print("keine x-werte")
    min_x = np.min(x_values)
    max_x = np.max(x_values)
    controlpoints = []
    i = 0
    while i <= max_x - min_x:                                   #getting lower edge of layer
        linie = min_x + i
        j = 0
        temp_linie = []
        while j < len(coord_data):
            if coord_data[j][0] == linie:
                temp_linie.append(coord_data[j])
                j += 1
            else:
                j += 1
                next
        if len(temp_linie) < 2:
            next
        else:            
            next
            min_index = np.argmin(temp_linie[:][1])
            controlpoints.append(temp_linie[min_index - 1])
        i += 1
    
    i = 0
    while i <= max_x - min_x:                                   #getting lower edge of layer
        linie = max_x - i
        j = 0
        temp_linie = []
        while j < len(coord_data):
            if coord_data[j][0] == linie:
                temp_linie.append(coord_data[j])
                j += 1
            else:
                j += 1
                next
        if len(temp_linie) < 2:
            next
        else:            
            next
            max_index = np.argmax(temp_linie[:][1])
            controlpoints.append(temp_linie[max_index - 1])
        i += 1
        
    return controlpoints

def plot_points(controlpoints, coord_data = []):
    "no londer in use"

    controlpoints_x_values_local = []
    controlpoints_y_values_local = []
    controlpoints_z_values_local = []
    x_values_local = []
    y_values_local = []
    z_values_local = []

    for line in controlpoints:                                  #controlpoints split by axis
        controlpoints_x_values_local.append(float(line[0]))         
        controlpoints_y_values_local.append(float(line[1]))
        controlpoints_z_values_local.append(float(line[2]))
    if any(coord_data) == True:
        for line in coord_data:
            x_values_local.append(float(line[0]))                 #coordinates split by axis
            y_values_local.append(float(line[1]))
            z_values_local.append(float(line[2]))
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if any(coord_data) == True:
        ax.scatter(x_values_local, y_values_local, z_values_local, marker = "d", s = 0.5)
    ax.scatter(controlpoints_x_values_local, controlpoints_y_values_local, controlpoints_z_values_local, color = "r")
    ax.set_xlabel("x-Axis")
    ax.set_ylabel("y-Axis")
    ax.set_zlabel("z-Axis")
    print("test")
    plt.show()
    return

def data_for_cylinder_along_z(center_x, center_y, x_radius, y_radius, height_z):
    "no londer in use"

    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = x_radius*np.cos(theta_grid) + center_x
    y_grid = y_radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

def data_for_cylinder_along_direction(center_a, center_b, a_radius, b_radius, start, end, direction, nbr_layers = 50, pts_layer = 50):
    """
    Returns x, y and z coordinates of the cyclinder

    Parameter
    ---------
    center_a : 1d array
        x, y OR z coordinate of the starting point of the cyclinder
    center_b : 1d array
        x, y OR z coordinate of the starting point of the cyclinder
            center_a and center_b make a 2d coordinate of the starting point
    a_radius : float
        radius a of the ellipse, calculated from the area
    b_radius : float
        radius b of the ellipse, calculated from the area
    start : float
        remaining axis start point
        e.g. if center_a gave an x-coordinate and center_b gave a y-coordinate then start (and end) must be z-coordinates
    direction : string
        must be "x", "y" or "z", discribe the axis along which the cylinder is generated
    nbr_layers : int
        number of layers that the generated data shoud contain
        Default is 50
    pts_layer : int
        number of points per layer that the generated data shoud contain

    Returns
    -------
    x_grid : 1d array
        x-coordinates of the hull of the cylinder
    y_grid : 1d array
        y-coordinates of the hull of the cylinder
    z_grid : 1d array
        z-coordinates of the hull of the cylinder

    should be used in the ax.plot_surface function of matplotlib
    """
    if direction == "z":
        z = np.linspace(start, end, nbr_layers)
        theta = np.linspace(0, 2*np.pi, pts_layer)
        theta_grid, z_grid=np.meshgrid(theta, z)
        x_grid = a_radius*np.cos(theta_grid) + center_a
        y_grid = b_radius*np.sin(theta_grid) + center_b
    elif direction == "y":
        z = np.linspace(start, end, nbr_layers)
        theta = np.linspace(0, 2*np.pi, pts_layer)
        theta_grid, y_grid=np.meshgrid(theta, z)
        x_grid = a_radius*np.cos(theta_grid) + center_a
        z_grid = b_radius*np.sin(theta_grid) + center_b
    elif direction == "x":
        z = np.linspace(start, end, nbr_layers)
        theta = np.linspace(0, 2*np.pi, pts_layer)
        theta_grid, x_grid=np.meshgrid(theta, z)
        y_grid = a_radius*np.cos(theta_grid) + center_a
        z_grid = b_radius*np.sin(theta_grid) + center_b
    return x_grid,y_grid,z_grid

def get_hull_points(layer_points, level):
    """
    no londer in use 


    Return a 2d array of coordinates of hull points

    Parameter
    ---------
    layer_points : 2d array
        coordniates of all points in a layer, e.g., [[x1 = 1.3, y1 = 2.54, z1 = 2.31][...]]
    level : string
        plane of the layer, e.g. layer in y-z-plane --> level = "x"
        the coordinates given in layer_points must all have the same value for the level dimension, 
        e.g., level = "x" --> all x same value

    Returns
    -------
    hull_points : 2d array
        coordinates of the hull points, e.g., [[x1 = 1.1, y1 = 4.14, z1 = 9.21][...]]
    """
    if level == "x":
        a = 1
        b = 2
    elif level == "y":
        a = 0 
        b = 2
    elif level == "z":
        a = 0
        b = 1
    else:
        print("level hast to be x,y or z")
    layer_points_2d = []
    for line in layer_points:                                  #controlpoints split by axis
        coord = []
        coord.append(float(line[a]))
        coord.append(float(line[b]))
        layer_points_2d.append(coord)
    hull = ConvexHull(layer_points_2d)
    hull_points = []
    line = 0
    while line < len(hull.vertices) - 1:
        hull_points.append(layer_points[hull.vertices[line]])
        line += 1
    return hull_points

def get_edge_controlpoints(coord_data, axis):
    """
    Return a 2d array of coordinates of edge points

    Parameter
    ---------
    layer_points : 2d array
        coordniates of all points in a layer, e.g., [[x1 = 1.3, y1 = 2.54, z1 = 2.31][...]]
    axis : string
        axis on which the cylinder lies
        the coordinates given in coord_data must all have the same value for the axis dimension, 
        e.g., axis = "x" --> all x same value

    Returns
    -------
    controlpoints : 2d array
        coordinates of the edge points, e.g., [[x1 = 1.1, y1 = 4.14, z1 = 9.21][...]]
    """
    controlpoints = []
    x_values = []
    y_values = []
    z_values = []
    for line in coord_data:
        x_values.append(float(line[0]))
        y_values.append(float(line[1]))
        z_values.append(float(line[2]))

    if axis == "x":

        if np.max(y_values) - np.min(y_values) >= np.max(z_values) - np.min(z_values):
            start_line = np.min(y_values)
            end_line = np.max(y_values)
            min_max_line = 2
            line_axis = 1
        elif np.max(y_values) - np.min(y_values) < np.max(z_values) - np.min(z_values):
            start_line = np.min(z_values)
            end_line = np.max(z_values)
            min_max_line = 1
            line_axis = 2
            
        i = 0
        while i <= end_line - start_line:                                   #getting lower edge of layer
            line = start_line + i
            j = 0
            temp_line = []
            while j < len(coord_data):
                if coord_data[j][line_axis] == line:
                    temp_line.append(coord_data[j])
                    j += 1
                else:
                    j += 1
                    continue
            if len(temp_line) < 2:
                i += 1
                continue
            else:
                temp_line = np.array(temp_line)
                min_index = np.argmin(temp_line[:, min_max_line])
                controlpoints.append(temp_line[min_index])
                i += 1

        i = 0
        while i <= end_line - start_line:                                   #getting lower edge of layer
            line = end_line - i
            j = 0
            temp_line = []
            while j < len(coord_data):
                if coord_data[j][line_axis] == line:
                    temp_line.append(coord_data[j])
                    j += 1
                else:
                    j += 1
                    continue
            if len(temp_line) < 2:
                i += 1
                continue
            else:
                temp_line = np.array(temp_line)
                max_index = np.argmax(temp_line[:, min_max_line])
                controlpoints.append(temp_line[max_index])
                i += 1
        
    if axis == "y":

        if np.max(x_values) - np.min(x_values) >= np.max(z_values) - np.min(z_values):
            start_line = np.min(x_values)
            end_line = np.max(x_values)
            min_max_line = 2
            line_axis = 0
        elif np.max(x_values) - np.min(x_values) < np.max(z_values) - np.min(z_values):
            start_line = np.min(z_values)
            end_line = np.max(z_values)
            min_max_line = 0
            line_axis = 2

        i = 0
        while i <= end_line - start_line:                                   #getting lower edge of layer
            line = start_line + i
            j = 0
            temp_line = []
            while j < len(coord_data):
                if coord_data[j][line_axis] == line:
                    temp_line.append(coord_data[j])
                    j += 1
                else:
                    j += 1
                    continue
            if len(temp_line) < 2:
                i += 1
                continue
            else:
                temp_line = np.array(temp_line)
                min_index = np.argmin(temp_line[:, min_max_line])
                controlpoints.append(temp_line[min_index])
                i += 1
        
        i = 0
        while i <= end_line - start_line:                                   #getting lower edge of layer
            line = end_line - i
            j = 0
            temp_line = []
            while j < len(coord_data):
                if coord_data[j][line_axis] == line:
                    temp_line.append(coord_data[j])
                    j += 1
                else:
                    j += 1
                    continue
            if len(temp_line) < 2:
                i += 1
                continue
            else:
                temp_line = np.array(temp_line)
                max_index = np.argmax(temp_line[:, min_max_line])
                controlpoints.append(temp_line[max_index])
                i += 1

    if axis == "z":

        if np.max(x_values) - np.min(x_values) >= np.max(y_values) - np.min(y_values):
            start_line = np.min(x_values)
            end_line = np.max(x_values)
            min_max_line = 1
            line_axis = 0
        elif np.max(x_values) - np.min(x_values) < np.max(y_values) - np.min(y_values):
            start_line = np.min(y_values)
            end_line = np.max(y_values)
            min_max_line = 0
            line_axis = 1
        i = 0

        while i <= end_line - start_line:                                   #getting lower edge of layer
            line = start_line + i
            j = 0
            temp_line = []
            while j < len(coord_data):
                if coord_data[j][line_axis] == line:
                    temp_line.append(coord_data[j])
                    j += 1
                else:
                    j += 1
                    continue
            if len(temp_line) < 2:
                i += 1
                continue
            else:
                temp_line = np.array(temp_line)
                min_index = np.argmin(temp_line[:, min_max_line])
                controlpoints.append(temp_line[min_index])
                i += 1
        
        i = 0
        while i <= end_line - start_line:                                   #getting upper edge of layer
            line = end_line - i
            j = 0
            temp_line = []
            while j < len(coord_data):
                if coord_data[j][line_axis] == line:
                    temp_line.append(coord_data[j])
                    j += 1
                else:
                    j += 1
                    continue
            if len(temp_line) < 2:
                i += 1
                continue
            else:
                temp_line = np.array(temp_line)
                max_index = np.argmax(temp_line[:, min_max_line])
                controlpoints.append(temp_line[max_index])
                i += 1
    return controlpoints


def export_points_txt(x, y, z, name):
    """
    exports the given points to a .txt file and saves it in the same folder in which the program is located

    Parameter
    ---------
    x : 1d array
        x coordinates of points
    y : 1d array
        y coordinates of points
    z : 1d array
        z coordinates of points
    name : string
        name of the .txt file

    Returns
    -------
    nothing
    """


    nametxt = name + ".txt"
    f = open(nametxt, "w+")

    i = 0
    while i < len(x):
        j = 0
        while j < len(x[i]):
            f.write(str(x[i][j]))
            f.write(",")
            f.write(str(y[i][j]))
            f.write(",")
            f.write(str(z[i][j]))
            f.write("\n")
            j += 1
        i += 1
    f.close()
    return

def export_points_txt_append(x, y, z, name):
    """
    adds more points to the created .txt file from the export_points_txt function
    can only be run after the export_points_txt function
    name has to be identical

    Parameter
    ---------
    x : 1d array
        x coordinates of points
    y : 1d array
        y coordinates of points
    z : 1d array
        z coordinates of points
    name : string
        name of the .txt file

    Returns
    -------
    nothing
    """
    nametxt = name + ".txt"
    f = open(nametxt, "a")

    i = 0
    while i < len(x):
        j = 0
        while j < len(x[i]):
            f.write(str(x[i][j]))
            f.write(",")
            f.write(str(y[i][j]))
            f.write(",")
            f.write(str(z[i][j]))
            f.write("\n")
            j += 1
        i += 1
    f.close()
    return

def closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):
    """
    Return the closest distance between two lines
    The lines are defined by two points

    Parameter
    ---------
    a0 : 1d array
        coordniates of point 1 on first line
    a1 : 1d array
        coordniates of point 2 on first line
    b0 : 1d array
        coordniates of point 1 on second line
    b1 : 1d array
        coordniates of point 2 on second line

    Returns
    -------
    hull_points : 2d array
        coordinates of the hull points, e.g., [[x1 = 1.1, y1 = 4.14, z1 = 9.21][...]]
    pA : 1d array
        closest point on first line
    pB : 1d array
        closest point on second line
    np.linalg.norm(pA-pB) : float
        distance between lines
    """

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0=True
        clampA1=True
        clampB0=True
        clampB1=True


    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
    
    _A = A / magA
    _B = B / magB
    
    cross = np.cross(_A, _B);
    denom = np.linalg.norm(cross)**2
    
    
    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A,(b0-a0))
        
        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A,(b1-a0))
            
            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0,b0,np.linalg.norm(a0-b0)
                    return a0,b1,np.linalg.norm(a0-b1)
                
                
            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1,b0,np.linalg.norm(a1-b0)
                    return a1,b1,np.linalg.norm(a1-b1)
                
                
        # Segments overlap, return distance between parallel segments
        return None,None,np.linalg.norm(((d0*_A)+a0)-b0)
        
    
    
    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0);
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom;
    t1 = detB/denom;

    pA = a0 + (_A * t0) # Projected closest point on segment A
    pB = b0 + (_B * t1) # Projected closest point on segment B


    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1
        
        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1
            
        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B,(pA-b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)
    
        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A,(pB-a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    
    return pA,pB,np.linalg.norm(pA-pB)





