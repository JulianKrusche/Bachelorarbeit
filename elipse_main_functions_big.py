import csv
import operator
from operator import itemgetter
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as pltp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from scipy.interpolate import BSpline
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import Polygon
import sys 

from elipse_main_functions import get_medium
from elipse_main_functions import max_point
from elipse_main_functions import get_edge_controlpoints
from elipse_main_functions import plot_points
from elipse_main_functions import data_for_cylinder_along_direction
from elipse_main_functions import data_for_cylinder_along_z
from elipse_main_functions import get_hull_points
from elipse_main_functions import closestDistanceBetweenLines
from elipse_main_functions import plypolygon


def ellipse_along_axis(coord_data, axis, set_number_of_layers = 10, ply = None):
    """
    calculates the parameters for the elliptic cylinder

    Parameter
    ---------
    coord_data : 2d array
        coordinates of all points, e.g., [[x1 = 1.3, y1 = 2.54, z1 = 2.31][...]]
    axis : string
        axis ("x", "y" or "z") to which fibre is parallel
    set_number_of_layers
        number of layers of with the area is calculated
        more layers will take noticeable longer 

    Returns
    -------
    axis1 : string
        one letter string containing "x", "y" or "z"
        describes the axis to which the big fibre is parallel  
    pfp_x_values : 1d array
        x-coordinates of the edge points
    pfp_y_values : 1d array
        y-coordinates of the edge points
    pfp_z_values : 1d array
        y-coordinates of the edge points
    Xc : 2d numpy array
        x-coordinates for plotting the cylinder split by layer
    Yc : 2d numpy array
        y-coordinates for plotting the cylinder split by layer
    Zc : 2d numpy array
        z-coordinates for plotting the cylinder split by layer
    height : float
        height of the base ellipse of the cylinder
    width : float
        width of the base ellipse of the cylinder
    lenght : float
        lenght of the cylinder
    """
    if axis == "x":
        nbr_axis = 0
    elif axis == "y":
        nbr_axis = 1
    elif axis == "z":
        nbr_axis = 2

    coord_data_sorted = sorted(coord_data, key=operator.itemgetter(nbr_axis), reverse=False)

    pfp_x_values = []
    pfp_y_values = []
    pfp_z_values = []
    area_sum = 0
    start_layer = int(min(coord_data_sorted, key=itemgetter(nbr_axis))[nbr_axis]) 
    end_layer = int(max(coord_data_sorted, key=itemgetter(nbr_axis))[nbr_axis]) 
    skip_layer = int((end_layer - start_layer)  / set_number_of_layers)
    number_layers = 0
    layer_medium_a = 0
    layer_medium_b = 0
    max_distance = 0
    layer_medium_abc = [0, 0, 0]
    for layer in range(start_layer + skip_layer, end_layer - skip_layer, skip_layer):
        controlpoints_x_values = []
        controlpoints_y_values = []
        controlpoints_z_values = []
        coord_data = []
        x_values = []
        y_values = []
        z_values = []
        safety = False
        points_in_layer = 0
        number_layers += 1
        layer_medium_abc[nbr_axis] = layer
        for line in coord_data_sorted:
            if float(line[nbr_axis]) == layer:
                points_in_layer += 1
                x_values.append(float(line[0]))                 #coordinates split by axis
                y_values.append(float(line[1]))
                z_values.append(float(line[2]))
                coord = []
                coord_data_pre = [0, 0, 0]
                coord.append(float(line[0]))                    #coordinates combined
                coord.append(float(line[1]))
                coord.append(float(line[2]))
                coord_data_pre = [float(line[0]), float(line[1]), float(line[2])]
                if number_layers > 1:
                    if math.dist(coord_data_pre, layer_medium_abc) > 2 * max_distance:
                        safety = True
                        break 
                coord_data.append(coord)
            elif float(line[nbr_axis]) > layer:
                break
            else:
                continue

        if safety == True:
            number_layers -= 1
            continue

        if axis == "x":
            if number_layers == 1:
                layer_medium_a = get_medium(y_values)
                layer_medium_b = get_medium(z_values)
            else:
                layer_medium_a = (layer_medium_a + get_medium(y_values)) / 2
                layer_medium_b = (layer_medium_b + get_medium(z_values)) / 2
            layer_medium_abc[1] = layer_medium_a
            layer_medium_abc[2] = layer_medium_b
        
        if axis == "y":
            if number_layers == 1:
                layer_medium_a = get_medium(x_values)
                layer_medium_b = get_medium(z_values)
            else:
                layer_medium_a = (layer_medium_a + get_medium(x_values)) / 2
                layer_medium_b = (layer_medium_b + get_medium(z_values)) / 2
            layer_medium_abc[0] = layer_medium_a
            layer_medium_abc[2] = layer_medium_b
        
        if axis == "z":
            if number_layers == 1:
                layer_medium_a = get_medium(x_values)
                layer_medium_b = get_medium(y_values)
            else:
                layer_medium_a = (layer_medium_a + get_medium(x_values)) / 2
                layer_medium_b = (layer_medium_b + get_medium(y_values)) / 2
            layer_medium_abc[0] = layer_medium_a
            layer_medium_abc[1] = layer_medium_b

        for line in coord_data:
            if math.dist(line, layer_medium_abc) > max_distance:
                max_distance = math.dist(line, layer_medium_abc)

        if ply == False:
            controlpoints = get_edge_controlpoints(coord_data, axis)

            for line in controlpoints:                                  #controlpoints split by axis
                controlpoints_x_values.append(float(line[0]))         
                controlpoints_y_values.append(float(line[1]))
                controlpoints_z_values.append(float(line[2]))
                pfp_x_values.append(float(line[0]))         
                pfp_y_values.append(float(line[1]))
                pfp_z_values.append(float(line[2]))

        polypoints = plypolygon(coord_data)

        for line in polypoints:                                  #controlpoints split by axis
            controlpoints_x_values.append(float(line[0]))         
            controlpoints_y_values.append(float(line[1]))
            controlpoints_z_values.append(float(line[2]))
            pfp_x_values.append(float(line[0]))         
            pfp_y_values.append(float(line[1]))
            pfp_z_values.append(float(line[2]))

        if axis == "x":
            layer_area = Polygon(zip(controlpoints_y_values, controlpoints_z_values)) # area of layer
            area_sum += float(layer_area.area)
        elif axis == "y":
            layer_area = Polygon(zip(controlpoints_x_values, controlpoints_z_values)) # area of layer
            area_sum += float(layer_area.area)
        elif axis == "z":
            layer_area = Polygon(zip(controlpoints_y_values, controlpoints_x_values)) # area of layer
            area_sum += float(layer_area.area)


    area = area_sum / number_layers

    if axis == "x":
        width_pre = np.max(pfp_y_values) - np.min(pfp_y_values)
        height_pre = np.max(pfp_z_values) - np.min(pfp_z_values)
        w_h_ratio = width_pre / height_pre
        a_radius = math.sqrt(area / ((1/w_h_ratio) * math.pi))
        b_radius = a_radius / w_h_ratio 
        Xc,Yc,Zc = data_for_cylinder_along_direction(get_medium(pfp_y_values), get_medium(pfp_z_values), a_radius, b_radius, start_layer, end_layer, axis)
    elif axis == "y":
        width_pre = np.max(pfp_x_values) - np.min(pfp_x_values)
        height_pre = np.max(pfp_z_values) - np.min(pfp_z_values)
        w_h_ratio = width_pre / height_pre
        a_radius = math.sqrt(area / ((1/w_h_ratio) * math.pi))
        b_radius = a_radius / w_h_ratio 
        Xc,Yc,Zc = data_for_cylinder_along_direction(get_medium(pfp_x_values), get_medium(pfp_z_values), a_radius, b_radius, start_layer, end_layer, axis)
    elif axis == "z":
        width_pre = np.max(pfp_x_values) - np.min(pfp_x_values)
        height_pre = np.max(pfp_y_values) - np.min(pfp_y_values)
        w_h_ratio = width_pre / height_pre
        a_radius = math.sqrt(area / ((1/w_h_ratio) * math.pi))
        b_radius = a_radius / w_h_ratio 
        Xc,Yc,Zc = data_for_cylinder_along_z(get_medium(pfp_x_values), get_medium(pfp_y_values), a_radius, b_radius, end_layer)

    if a_radius > b_radius:
        height = b_radius
        width = a_radius
    else:
        height = a_radius
        width = b_radius

    length = end_layer - start_layer

    return pfp_x_values, pfp_y_values, pfp_z_values, Xc, Yc, Zc, 2 * height, 2 * width, length


def check_axis(coord_data):
    """
    checks along with axis the fibre lies

    Parameter
    ---------
    coord_data : 2d array
        coordinates of all points, e.g., [[x1 = 1.3, y1 = 2.54, z1 = 2.31][...]]

    Returns
    -------
    axis1 : string
        one letter string containing "x", "y" or "z"
        describes the axis to which the big fibre is parallel  
    axis2 : string
        one letter string containing "x", "y" or "z"
        describes the axis to which the small fibre is parallel 
    datamean : 1d array
        coordinate of the mean of all points 
    scale : float
        scaling distance for plotting the data
    linedistance : float
        distance between the two lines from singular value decomposition
    """
    yes = ['yes','y', 'ye']
    no = ['no','n']
    coord_data_short = []
    pfp_x_values = []
    pfp_y_values = []
    pfp_z_values = []
    coord = []
    i = 0
    j= 0
    while j < len(coord_data):
        if i > len(coord_data) / 2000:
            coord.append(float(coord_data[j][0]))                    #coordinates combined
            coord.append(float(coord_data[j][1]))
            coord.append(float(coord_data[j][2]))
            pfp_x_values.append(float(coord_data[j][0]))         
            pfp_y_values.append(float(coord_data[j][1]))
            pfp_z_values.append(float(coord_data[j][2]))
            coord_data_short.append(coord)
            coord = []
            i = 0
        else:
            i += 1
        j += 1
    np_coord_data = np.array(coord_data_short)
    datamean = np_coord_data.mean(axis=0)
    scale = float(np.max([[datamean[0] - np.min(pfp_x_values)], [datamean[1] - np.min(pfp_y_values)], [datamean[2] - np.min(pfp_z_values)], [np.max(pfp_x_values) - datamean[0]], [np.max(pfp_y_values) - datamean[1]], [np.max(pfp_z_values) - datamean[2]]]))
    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(coord_data_short - datamean)
    #print("Vector big axis:", vv[0])
    x_vector = [1, 0, 0]
    y_vector = [0, 1, 0]
    z_vector = [0, 0, 1]

    if abs(vv[0][0]) - 1 < 0.05 and abs(vv[0][1]) < 0.05 and abs(vv[0][2]) < 0.05:
        axis1 = "x"
    elif abs(vv[0][0]) < 0.05 and abs(vv[0][1]) - 1 < 0.05 and abs(vv[0][2]) < 0.05:
        axis1 = "y"
    elif abs(vv[0][0]) < 0.05 and abs(vv[0][1]) < 0.05 and abs(vv[0][2]) - 1 < 0.05:
        axis1 = "z"
    else:
        print("Vector big axis:", vv[0])
        linepts = vv[0] * np.mgrid[-1200:1200:2j][:, np.newaxis]
        linepts += datamean
        x_line = x_vector * np.mgrid[float(np.min(pfp_x_values) - datamean[0]) - 100:float(np.max(pfp_x_values) - datamean[0]) + 100:2j][:, np.newaxis]
        x_line += datamean
        y_line = y_vector * np.mgrid[float(np.min(pfp_y_values) - datamean[1]) - 100:float(np.max(pfp_y_values) - datamean[1]) + 100:2j][:, np.newaxis]
        y_line += datamean
        z_line = z_vector * np.mgrid[float(np.min(pfp_z_values) - datamean[2]) - 100:float(np.max(pfp_z_values) - datamean[2]) + 100:2j][:, np.newaxis]
        z_line += datamean
        major_ticks = np.arange(-500, 2500, 500)
        minor_ticks = np.arange(-1000, 2000, 500)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pfp_x_values, pfp_y_values, pfp_z_values, s = 0.2)
        ax.plot(*linepts.T, c = "red", label = "This line should be nearly identical \nto one of the blue lines")
        ax.plot(*x_line.T, c = "blue", label = "axis lines")
        ax.plot(*y_line.T, c = "blue")
        ax.plot(*z_line.T, c = "blue")
        ax.set_xlabel("x-Axis")
        ax.set_ylabel("y-Axis")
        ax.set_zlabel("z-Axis")
        ax.set_xlim3d(float(datamean[0]) - scale, float(datamean[0]) + scale)
        ax.set_ylim3d(float(datamean[1]) - scale, float(datamean[1]) + scale)
        ax.set_zlim3d(float(datamean[2]) - scale, float(datamean[2]) + scale)
        ax.set_xticks(major_ticks)
        ax.set_yticks(np.arange(0, 2500, 500))
        ax.set_zticks(minor_ticks)
        ax.legend(fontsize = "10")
        plt.show()
        sys.exit("Fibre data has to be along axis")

    if axis1 == "x":
        min_x = np.min(pfp_x_values)
        max_x = np.max(pfp_x_values)
        dif = max_x - min_x

        h = 0
        coord = []
        coord_data_calax2 = []
        while h < len(coord_data_short):
            if (coord_data_short[h][0] > min_x + dif * 0.1 and coord_data_short[h][0] < min_x + dif * 0.35) or (coord_data_short[h][0] > min_x + dif * 0.65 and coord_data_short[h][0] < min_x + dif * 0.9):
                coord.append(float(coord_data_short[h][0]))
                coord.append(float(coord_data_short[h][1]))
                coord.append(float(coord_data_short[h][2]))
                coord_data_calax2.append(coord)
                coord = []
                h += 1
            else:
                h += 1
                
        x_values_local = []
        y_values_local = []
        z_values_local = []
        for line in coord_data_calax2:
            x_values_local.append(float(line[0]))                 #coordinates split by axis
            y_values_local.append(float(line[1]))
            z_values_local.append(float(line[2]))
        
        border_y_low = np.min(y_values_local)
        border_y_high = np.max(y_values_local)
        border_z_low = np.min(z_values_local)
        border_z_high = np.max(z_values_local)

        pfp_x_values_shorter = []
        pfp_y_values_shorter = []
        pfp_z_values_shorter = []
        h = 0
        coord = []
        coord_data_shorter = []
        while h < len(coord_data_short):
            if (coord_data_short[h][1] < border_y_low or coord_data_short[h][1] > border_y_high or coord_data_short[h][2] < border_z_low or coord_data_short[h][2] > border_z_high) and coord_data_short[h][0] > min_x + dif * 0.35 and coord_data_short[h][0] < min_x + dif * 0.65:
                coord.append(float(coord_data_short[h][0]))                    #coordinates combined
                coord.append(float(coord_data_short[h][1]))
                coord.append(float(coord_data_short[h][2]))
                coord_data_shorter.append(coord)
                coord = []
                pfp_x_values_shorter.append(float(coord_data_short[h][0]))                    #coordinates combined
                pfp_y_values_shorter.append(float(coord_data_short[h][1]))
                pfp_z_values_shorter.append(float(coord_data_short[h][2]))
                h += 1
            else:
                h += 1

        coord_data_not_in_shorter = []
        coord_pre = []
        pfp_x_values2 = []
        pfp_y_values2 = []
        pfp_z_values2 = []
        for line in coord_data_short:
            if line  not in coord_data_shorter:
                coord_pre.append(float(line[0]))                    #coordinates combined
                coord_pre.append(float(line[1]))
                coord_pre.append(float(line[2]))
                coord_data_not_in_shorter.append(coord_pre)
                coord_pre = []
                pfp_x_values2.append(float(line[0]))         
                pfp_y_values2.append(float(line[1]))
                pfp_z_values2.append(float(line[2]))

        #########Grenzwerte
        min_y = np.min(pfp_y_values)
        max_y = np.max(pfp_y_values)
        min_z = np.min(pfp_z_values)
        max_z = np.max(pfp_z_values)
        l2max = np.max([[max_y - min_y], [max_z - min_z]])
        lrealation = float(dif) / float(l2max)
        #print("Grenzwert:", 0.05 * lrealation)
        vector_limit = 0.05 * lrealation
        #########
        np_coord_data2 = np.array(coord_data_shorter)
        datamean2 = np_coord_data2.mean(axis=0)
        uu2, dd2, vv2 = np.linalg.svd(coord_data_shorter - datamean2)
        #print("Vector small axis:", vv2[0])

        if abs(vv2[0][0]) < vector_limit and abs(vv2[0][1]) - 1 < vector_limit and abs(vv2[0][2]) < vector_limit:
            axis2 = "y"
        elif abs(vv2[0][0]) < vector_limit and abs(vv2[0][1]) < vector_limit and abs(vv2[0][2]) - 1 < vector_limit:
            axis2 = "z"
        else:
            print("Vector small axis:", vv2[0])
            x_values_local = []
            y_values_local = []
            z_values_local = []
            linepts2 = vv2[0] * np.mgrid[-1500:1500:2j][:, np.newaxis]
            linepts2 += datamean2
            for line in coord_data_shorter:
                x_values_local.append(float(line[0]))                 #coordinates split by axis
                y_values_local.append(float(line[1]))
                z_values_local.append(float(line[2]))
            linepts2 = vv2[0] * np.mgrid[-1500:1500:2j][:, np.newaxis]
            linepts2 += datamean2
            x_line = x_vector * np.mgrid[float(np.min(pfp_x_values) - datamean2[0]) - 100:float(np.max(pfp_x_values) - datamean[0]) + 100:2j][:, np.newaxis]
            x_line += datamean2
            y_line = y_vector * np.mgrid[float(np.min(pfp_y_values) - datamean2[1]) - 100:float(np.max(pfp_y_values) - datamean[1]) + 100:2j][:, np.newaxis]
            y_line += datamean2
            z_line = z_vector * np.mgrid[float(np.min(pfp_z_values) - datamean2[2]) - 100:float(np.max(pfp_z_values) - datamean[2]) + 100:2j][:, np.newaxis]
            z_line += datamean2
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pfp_x_values, pfp_y_values, pfp_z_values, s = 0.2)
            ax.scatter(x_values_local, y_values_local, z_values_local, s = 0.5, c = "r")
            ax.plot(*linepts2.T, c = "red", label = "This line should be nearly identical \nto one of the blue lines")
            ax.plot(*x_line.T, c = "blue", label = "axis lines")
            ax.plot(*y_line.T, c = "blue")
            ax.plot(*z_line.T, c = "blue")
            ax.set_xlabel("x-Axis")
            ax.set_ylabel("y-Axis")
            ax.set_zlabel("z-Axis")
            ax.set_xlim3d(float(datamean[0]) - scale, float(datamean[0]) + scale)
            ax.set_ylim3d(float(datamean[1]) - scale, float(datamean[1]) + scale)
            ax.set_zlim3d(float(datamean[2]) - scale, float(datamean[2]) + scale)
            ax.legend(fontsize = "6")
            plt.show()
            user_input2 = input('Second fibre not along axis. Do you want to continue (result may be incorrect)? \nyes/y = continue \nno/n = stop')
            if user_input2.lower() in yes:
                print("continuing")
                pass
            elif user_input2.lower() in no:
                sys.exit("Program was stopped")
            else:
                sys.stdout.write("Please respond with 'yes' or 'no'")
        
        min_x = np.min(pfp_x_values)
        w = (min_x - datamean[0]) / vv[0][0]
        p1 = datamean + w * vv[0]

        max_x = np.max(pfp_x_values)
        e = (max_x - datamean[0]) / vv[0][0]
        p2 = datamean + e * vv[0]

        px = [p1[0], p2[0]]
        py = [p1[1], p2[1]]
        pz = [p1[2], p2[2]]

        distance_all = []
        for line in coord_data_not_in_shorter:
            p3 = line
            d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
            distance_all.append(d)
        distance_mean_1 = np.mean(distance_all)

        if axis2 == "y":
            min_y_2 = np.min(pfp_y_values_shorter)
            w_2 = (min_y_2 - datamean2[1]) / vv2[0][1]
            p1_2 = datamean2 + w_2 * vv2[0]

            max_y_2 = np.max(pfp_y_values_shorter)
            e_2 = (max_y_2 - datamean2[1]) / vv2[0][1]
            p2_2 = datamean2 + e_2 * vv2[0]
            px_2 = [p1_2[0], p2_2[0]]
            py_2 = [p1_2[1], p2_2[1]]
            pz_2 = [p1_2[2], p2_2[2]]
        elif axis2 == "z":
            min_z_2 = np.min(pfp_z_values_shorter)
            w_2 = (min_z_2 - datamean2[2]) / vv2[0][2]
            p1_2 = datamean2 + w_2 * vv2[0]

            max_z_2 = np.max(pfp_z_values_shorter)
            e_2 = (max_z_2 - datamean2[2]) / vv2[0][2]
            p2_2 = datamean2 + e_2 * vv2[0]

            px_2 = [p1_2[0], p2_2[0]]
            py_2 = [p1_2[1], p2_2[1]]
            pz_2 = [p1_2[2], p2_2[2]]

        distance_all2 = []
        for line in coord_data_shorter:
            p3 = line
            d = np.linalg.norm(np.cross(p2_2-p1_2, p1_2-p3))/np.linalg.norm(p2_2-p1_2)
            distance_all2.append(d)
        distance_mean_2 = np.mean(distance_all2)

        distance_mean = (distance_mean_1 * len(coord_data_short) + distance_mean_2 * len(coord_data_shorter)) / (len(coord_data_short)) 

        distance_scale = float(np.min([[datamean[0] - np.min(pfp_x_values)], [datamean[1] - np.min(pfp_y_values)], [datamean[2] - np.min(pfp_z_values)], [np.max(pfp_x_values) - datamean[0]], [np.max(pfp_y_values) - datamean[1]], [np.max(pfp_z_values) - datamean[2]]]))

        if distance_mean > distance_scale * 1.5:
            print("Distanz Durchschnitt:", distance_mean)
            print("Distanz scale:", distance_scale)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pfp_x_values, pfp_y_values, pfp_z_values, s = 0.2, label = "Points have an unusual high distance \nfrom the red line. To continue close the plot \nand decide in the terminal")
            ax.plot(px, py ,pz, c = "red")
            ax.plot(px_2, py_2 ,pz_2, c = "red")
            ax.set_xlabel("x-Axis")
            ax.set_ylabel("y-Axis")
            ax.set_zlabel("z-Axis")
            ax.set_xlim3d(float(datamean[0]) - scale, float(datamean[0]) + scale)
            ax.set_ylim3d(float(datamean[1]) - scale, float(datamean[1]) + scale)
            ax.set_zlim3d(float(datamean[2]) - scale, float(datamean[2]) + scale)
            ax.legend(fontsize = "6")
            plt.show()
            yes = ['yes','y', 'ye']
            no = ['no','n']
            user_input = input('Points have an unusual high distance \nfrom the red line. \nDo you want to continue? \nyes/y = continue \nno/n = stop\n')
            if user_input.lower() in yes:
                print("continuing")
                pass
            elif user_input.lower() in no:
                sys.exit("Program was stopped")
            else:
                sys.stdout.write("Please respond with 'yes' or 'no'")
    

    if axis1 == "y":
        min_y = np.min(pfp_y_values)
        max_y = np.max(pfp_y_values)
        dif = max_y- min_y

        h = 0
        coord = []
        coord_data_calax2 = []
        while h < len(coord_data_short):
            if (coord_data_short[h][1] > min_y + dif * 0.1 and coord_data_short[h][1] < min_y + dif * 0.35) or (coord_data_short[h][1] > min_y + dif * 0.65 and coord_data_short[h][1] < min_y + dif * 0.9):
                coord.append(float(coord_data_short[h][0]))
                coord.append(float(coord_data_short[h][1]))
                coord.append(float(coord_data_short[h][2]))
                coord_data_calax2.append(coord)
                coord = []
                h += 1
            else:
                h += 1
                
        x_values_local = []
        y_values_local = []
        z_values_local = []
        for line in coord_data_calax2:
            x_values_local.append(float(line[0]))                 #coordinates split by axis
            y_values_local.append(float(line[1]))
            z_values_local.append(float(line[2]))
        
        border_x_low = np.min(x_values_local)
        border_x_high = np.max(x_values_local)
        border_z_low = np.min(z_values_local)
        border_z_high = np.max(z_values_local)

        h = 0
        coord = []
        coord_data_shorter = []
        pfp_x_values_shorter = []
        pfp_y_values_shorter = []
        pfp_z_values_shorter = []
        while h < len(coord_data_short):
            if (coord_data_short[h][0] < border_x_low or coord_data_short[h][0] > border_x_high or coord_data_short[h][2] < border_z_low or coord_data_short[h][2] > border_z_high) and coord_data_short[h][1] > min_y + dif * 0.35 and coord_data_short[h][1] < min_y + dif * 0.65:
                coord.append(float(coord_data_short[h][0]))                    #coordinates combined
                coord.append(float(coord_data_short[h][1]))
                coord.append(float(coord_data_short[h][2]))
                coord_data_shorter.append(coord)
                coord = []
                pfp_x_values_shorter.append(float(coord_data_short[h][0]))                    #coordinates combined
                pfp_y_values_shorter.append(float(coord_data_short[h][1]))
                pfp_z_values_shorter.append(float(coord_data_short[h][2]))
                h += 1
            else:
                h += 1

        coord_data_not_in_shorter = []
        coord_pre = []
        pfp_x_values2 = []
        pfp_y_values2 = []
        pfp_z_values2 = []
        for line in coord_data_short:
            if line  not in coord_data_shorter:
                coord_pre.append(float(line[0]))                    #coordinates combined
                coord_pre.append(float(line[1]))
                coord_pre.append(float(line[2]))
                coord_data_not_in_shorter.append(coord_pre)
                coord_pre = []
                pfp_x_values2.append(float(line[0]))         
                pfp_y_values2.append(float(line[1]))
                pfp_z_values2.append(float(line[2]))

        #########Grenzwerte
        min_x = np.min(pfp_x_values)
        max_x = np.max(pfp_x_values)
        min_z = np.min(pfp_z_values)
        max_z = np.max(pfp_z_values)
        l2max = np.max([[max_x - min_x], [max_z - min_z]])
        lrealation = float(dif) / float(l2max)
        #print("Grenzwert:", 0.05 * lrealation)
        vector_limit = 0.05 * lrealation
        #########

        np_coord_data2 = np.array(coord_data_shorter)
        datamean2 = np_coord_data2.mean(axis=0)
        uu2, dd2, vv2 = np.linalg.svd(coord_data_shorter - datamean2)
        #print("Vector small axis:", vv2[0])        

        if abs(vv2[0][0]) - 1 < vector_limit and abs(vv2[0][1]) < vector_limit and abs(vv2[0][2]) < vector_limit:
            axis2 = "x"
        elif abs(vv2[0][0]) < vector_limit and abs(vv2[0][1]) < vector_limit and abs(vv2[0][2]) - 1 < vector_limit:
            axis2 = "z"
        else:
            print("Vector small axis:", vv2[0])
            linepts2 = vv2[0] * np.mgrid[-1500:1500:2j][:, np.newaxis]
            linepts2 += datamean
            x_line = x_vector * np.mgrid[float(np.min(pfp_x_values) - datamean[0]) - 100:float(np.max(pfp_x_values) - datamean[0]) + 100:2j][:, np.newaxis]
            x_line += datamean
            y_line = y_vector * np.mgrid[float(np.min(pfp_y_values) - datamean[1]) - 100:float(np.max(pfp_y_values) - datamean[1]) + 100:2j][:, np.newaxis]
            y_line += datamean
            z_line = z_vector * np.mgrid[float(np.min(pfp_z_values) - datamean[2]) - 100:float(np.max(pfp_z_values) - datamean[2]) + 100:2j][:, np.newaxis]
            z_line += datamean
            scale = float(np.max([[datamean[0] - np.min(pfp_x_values)], [datamean[1] - np.min(pfp_y_values)], [datamean[2] - np.min(pfp_z_values)], [np.max(pfp_x_values) - datamean[0]], [np.max(pfp_y_values) - datamean[1]], [np.max(pfp_z_values) - datamean[2]]]))
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pfp_x_values, pfp_y_values, pfp_z_values, s = 0.2)
            ax.plot(*linepts2.T, c = "red", label = "This line should be nearly identical \nto one of the blue lines")
            ax.plot(*x_line.T, c = "blue", label = "axis lines")
            ax.plot(*y_line.T, c = "blue")
            ax.plot(*z_line.T, c = "blue")
            ax.set_xlabel("x-Axis")
            ax.set_ylabel("y-Axis")
            ax.set_zlabel("z-Axis")
            ax.set_xlim3d(float(datamean[0]) - scale, float(datamean[0]) + scale)
            ax.set_ylim3d(float(datamean[1]) - scale, float(datamean[1]) + scale)
            ax.set_zlim3d(float(datamean[2]) - scale, float(datamean[2]) + scale)
            ax.legend(fontsize = "6")
            plt.show()
            user_input2 = input('Second fibre not along axis. Do you want to continue (result may be incorrect)? \nyes/y = continue \nno/n = stop')
            if user_input2.lower() in yes:
                print("continuing")
                pass
            elif user_input2.lower() in no:
                sys.exit("Program was stopped")
            else:
                sys.stdout.write("Please respond with 'yes' or 'no'")
        
        min_y = np.min(pfp_y_values)
        w = (min_y - datamean[1]) / vv[0][1]
        p1 = datamean + w * vv[0]

        max_y = np.max(pfp_y_values)
        e = (max_y - datamean[1]) / vv[0][1]
        p2 = datamean + e * vv[0]

        px = [p1[0], p2[0]]
        py = [p1[1], p2[1]]
        pz = [p1[2], p2[2]]

        distance_all = []
        for line in coord_data_not_in_shorter:
            p3 = line
            d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
            distance_all.append(d)
        distance_mean_1 = np.mean(distance_all)

        if axis2 == "x":
            min_x_2 = np.min(pfp_x_values_shorter)
            w_2 = (min_x_2 - datamean2[0]) / vv2[0][0]
            p1_2 = datamean2 + w_2 * vv2[0]

            max_x_2 = np.max(pfp_x_values_shorter)
            e_2 = (max_x_2 - datamean2[0]) / vv2[0][0]
            p2_2 = datamean2 + e_2 * vv2[0]
            px_2 = [p1_2[0], p2_2[0]]
            py_2 = [p1_2[1], p2_2[1]]
            pz_2 = [p1_2[2], p2_2[2]]
        elif axis2 == "z":
            min_z_2 = np.min(pfp_z_values_shorter)
            w_2 = (min_z_2 - datamean2[2]) / vv2[0][2]
            p1_2 = datamean2 + w_2 * vv2[0]

            max_z_2 = np.max(pfp_z_values_shorter)
            e_2 = (max_z_2 - datamean2[2]) / vv2[0][2]
            p2_2 = datamean2 + e_2 * vv2[0]

            px_2 = [p1_2[0], p2_2[0]]
            py_2 = [p1_2[1], p2_2[1]]
            pz_2 = [p1_2[2], p2_2[2]]

        distance_all2 = []
        for line in coord_data_shorter:
            p3 = line
            d = np.linalg.norm(np.cross(p2_2-p1_2, p1_2-p3))/np.linalg.norm(p2_2-p1_2)
            distance_all2.append(d)
        distance_mean_2 = np.mean(distance_all2)

        distance_mean = (distance_mean_1 * len(coord_data_short) + distance_mean_2 * len(coord_data_shorter)) / (len(coord_data_short)) 

        distance_scale = float(np.min([[datamean[0] - np.min(pfp_x_values)], [datamean[1] - np.min(pfp_y_values)], [datamean[2] - np.min(pfp_z_values)], [np.max(pfp_x_values) - datamean[0]], [np.max(pfp_y_values) - datamean[1]], [np.max(pfp_z_values) - datamean[2]]]))

        if distance_mean > distance_scale * 2:
            print("Distanz Durchschnitt:", distance_mean)
            print("Distanz scale:", distance_scale)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pfp_x_values, pfp_y_values, pfp_z_values, s = 0.2, label = "Points have an unusual high distance \nfrom the red line. To continue close the plot \nand decide in the terminal")
            ax.plot(px, py ,pz, c = "red")
            ax.plot(px_2, py_2 ,pz_2, c = "red")
            ax.set_xlabel("x-Axis")
            ax.set_ylabel("y-Axis")
            ax.set_zlabel("z-Axis")
            ax.set_xlim3d(float(datamean[0]) - scale, float(datamean[0]) + scale)
            ax.set_ylim3d(float(datamean[1]) - scale, float(datamean[1]) + scale)
            ax.set_zlim3d(float(datamean[2]) - scale, float(datamean[2]) + scale)
            ax.legend(fontsize = "6")
            plt.show()
            yes = ['yes','y', 'ye']
            no = ['no','n']
            user_input = input('Do you want to continue? \nyes/y = continue \nno/n = stop\n')
            if user_input.lower() in yes:
                print("continuing")
                pass
            elif user_input.lower() in no:
                sys.exit("Program was stopped")
            else:
                sys.stdout.write("Please respond with 'yes' or 'no'")
    

    if axis1 == "z":
        
        min_z = np.min(pfp_z_values)
        max_z = np.max(pfp_z_values)
        dif = max_z - min_z

        h = 0
        coord = []
        coord_data_calax2 = []
        while h < len(coord_data_short):
            if (coord_data_short[h][2] > min_z + dif * 0.1 and coord_data_short[h][2] < min_z + dif * 0.35) or (coord_data_short[h][2] > min_z + dif * 0.65 and coord_data_short[h][2] < min_z + dif * 0.9):
                coord.append(float(coord_data_short[h][0]))
                coord.append(float(coord_data_short[h][1]))
                coord.append(float(coord_data_short[h][2]))
                coord_data_calax2.append(coord)
                coord = []
                h += 1
            else:
                h += 1
                
        x_values_local = []
        y_values_local = []
        z_values_local = []
        for line in coord_data_calax2:
            x_values_local.append(float(line[0]))                 #coordinates split by axis
            y_values_local.append(float(line[1]))
            z_values_local.append(float(line[2]))
        
        border_x_low = np.min(x_values_local)
        border_x_high = np.max(x_values_local)
        border_y_low = np.min(y_values_local)
        border_y_high = np.max(y_values_local)

        h = 0
        coord = []
        coord_data_shorter = []
        pfp_x_values_shorter = []
        pfp_y_values_shorter = []
        pfp_z_values_shorter = []
        while h < len(coord_data_short):
            if (coord_data_short[h][0] < border_x_low or coord_data_short[h][0] > border_x_high or coord_data_short[h][1] < border_y_low or coord_data_short[h][1] > border_y_high) and coord_data_short[h][2] > min_z + dif * 0.35 and coord_data_short[h][2] < min_z + dif * 0.65:
                coord.append(float(coord_data_short[h][0]))                    #coordinates combined
                coord.append(float(coord_data_short[h][1]))
                coord.append(float(coord_data_short[h][2]))
                coord_data_shorter.append(coord)
                coord = []
                pfp_x_values_shorter.append(float(coord_data_short[h][0]))                    #coordinates combined
                pfp_y_values_shorter.append(float(coord_data_short[h][1]))
                pfp_z_values_shorter.append(float(coord_data_short[h][2]))
                h += 1
            else:
                h += 1

        coord_data_not_in_shorter = []
        coord_pre = []
        pfp_x_values2 = []
        pfp_y_values2 = []
        pfp_z_values2 = []
        for line in coord_data_short:
            if line  not in coord_data_shorter:
                coord_pre.append(float(line[0]))                    #coordinates combined
                coord_pre.append(float(line[1]))
                coord_pre.append(float(line[2]))
                coord_data_not_in_shorter.append(coord_pre)
                coord_pre = []
                pfp_x_values2.append(float(line[0]))         
                pfp_y_values2.append(float(line[1]))
                pfp_z_values2.append(float(line[2]))

        #########Grenzwerte
        min_x = np.min(pfp_x_values)
        max_x = np.max(pfp_x_values)
        min_y = np.min(pfp_y_values)
        max_y = np.max(pfp_y_values)
        l2max = np.max([[max_x - min_x], [max_y - min_y]])
        lrealation = float(dif) / float(l2max)
        #print("Grenzwert:", 0.05 * lrealation)
        vector_limit = 0.05 * lrealation
        #########

        np_coord_data2 = np.array(coord_data_shorter)
        datamean2 = np_coord_data2.mean(axis=0)
        uu2, dd2, vv2 = np.linalg.svd(coord_data_shorter - datamean2)
        #print("Vector small axis:", vv2[0])
        
        if abs(vv2[0][0]) - 1 < vector_limit and abs(vv2[0][1]) < vector_limit and abs(vv2[0][2]) < vector_limit:
            axis2 = "x"
        elif abs(vv2[0][0]) < vector_limit and abs(vv2[0][1]) - 1 < vector_limit and abs(vv2[0][2]) < vector_limit:
            axis2 = "y"
        else:
            print("Vector small axis:", vv2[0])
            linepts2 = vv2[0] * np.mgrid[-1500:1500:2j][:, np.newaxis]
            linepts2 += datamean
            x_line = x_vector * np.mgrid[float(np.min(pfp_x_values) - datamean[0]) - 100:float(np.max(pfp_x_values) - datamean[0]) + 100:2j][:, np.newaxis]
            x_line += datamean
            y_line = y_vector * np.mgrid[float(np.min(pfp_y_values) - datamean[1]) - 100:float(np.max(pfp_y_values) - datamean[1]) + 100:2j][:, np.newaxis]
            y_line += datamean
            z_line = z_vector * np.mgrid[float(np.min(pfp_z_values) - datamean[2]) - 100:float(np.max(pfp_z_values) - datamean[2]) + 100:2j][:, np.newaxis]
            z_line += datamean
            scale = float(np.max([[datamean[0] - np.min(pfp_x_values)], [datamean[1] - np.min(pfp_y_values)], [datamean[2] - np.min(pfp_z_values)], [np.max(pfp_x_values) - datamean[0]], [np.max(pfp_y_values) - datamean[1]], [np.max(pfp_z_values) - datamean[2]]]))
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pfp_x_values, pfp_y_values, pfp_z_values, s = 0.2)
            ax.plot(*linepts2.T, c = "red", label = "This line should be nearly identical \nto one of the blue lines")
            ax.plot(*x_line.T, c = "blue", label = "axis lines")
            ax.plot(*y_line.T, c = "blue")
            ax.plot(*z_line.T, c = "blue")
            ax.set_xlabel("x-Axis")
            ax.set_ylabel("y-Axis")
            ax.set_zlabel("z-Axis")
            ax.set_xlim3d(float(datamean[0]) - scale, float(datamean[0]) + scale)
            ax.set_ylim3d(float(datamean[1]) - scale, float(datamean[1]) + scale)
            ax.set_zlim3d(float(datamean[2]) - scale, float(datamean[2]) + scale)
            ax.legend(fontsize = "6")
            plt.show()
            user_input2 = input('Second fibre not along axis. Do you want to continue (result may be incorrect)? \nyes/y = continue \nno/n = stop')
            if user_input2.lower() in yes:
                print("continuing")
                pass
            elif user_input2.lower() in no:
                sys.exit("Program was stopped")
            else:
                sys.stdout.write("Please respond with 'yes' or 'no'")
        
        min_z = np.min(pfp_z_values)
        w = (min_z - datamean[2]) / vv[0][2]
        p1 = datamean + w * vv[0]

        max_z = np.max(pfp_z_values)
        e = (max_z - datamean[2]) / vv[0][2]
        p2 = datamean + e * vv[0]

        px = [p1[0], p2[0]]
        py = [p1[1], p2[1]]
        pz = [p1[2], p2[2]]

        distance_all = []
        for line in coord_data_not_in_shorter:
            p3 = line
            d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
            distance_all.append(d)
        distance_mean_1 = np.mean(distance_all)

        if axis2 == "x":
            min_x_2 = np.min(pfp_x_values_shorter)
            w_2 = (min_x_2 - datamean2[0]) / vv2[0][0]
            p1_2 = datamean2 + w_2 * vv2[0]

            max_x_2 = np.max(pfp_x_values_shorter)
            e_2 = (max_x_2 - datamean2[0]) / vv2[0][0]
            p2_2 = datamean2 + e_2 * vv2[0]
            px_2 = [p1_2[0], p2_2[0]]
            py_2 = [p1_2[1], p2_2[1]]
            pz_2 = [p1_2[2], p2_2[2]]
        if axis2 == "y":
            min_y_2 = np.min(pfp_y_values_shorter)
            w_2 = (min_y_2 - datamean2[1]) / vv2[0][1]
            p1_2 = datamean2 + w_2 * vv2[0]

            max_y_2 = np.max(pfp_y_values_shorter)
            e_2 = (max_y_2 - datamean2[1]) / vv2[0][1]
            p2_2 = datamean2 + e_2 * vv2[0]
            px_2 = [p1_2[0], p2_2[0]]
            py_2 = [p1_2[1], p2_2[1]]
            pz_2 = [p1_2[2], p2_2[2]]

        distance_all2 = []
        for line in coord_data_shorter:
            p3 = line
            d = np.linalg.norm(np.cross(p2_2-p1_2, p1_2-p3))/np.linalg.norm(p2_2-p1_2)
            distance_all2.append(d)
        distance_mean_2 = np.mean(distance_all2)

        distance_mean = (distance_mean_1 * len(coord_data_short) + distance_mean_2 * len(coord_data_shorter)) / (len(coord_data_short)) 

        distance_scale = float(np.min([[datamean[0] - np.min(pfp_x_values)], [datamean[1] - np.min(pfp_y_values)], [datamean[2] - np.min(pfp_z_values)], [np.max(pfp_x_values) - datamean[0]], [np.max(pfp_y_values) - datamean[1]], [np.max(pfp_z_values) - datamean[2]]]))

        if distance_mean > distance_scale * 2:
            print("Distanz Durchschnitt:", distance_mean)
            print("Distanz scale:", distance_scale)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pfp_x_values, pfp_y_values, pfp_z_values, s = 0.2, label = "Points have an unusual high distance \nfrom the red line. To continue close the plot \nand decide in the terminal")
            ax.plot(px, py ,pz, c = "red")
            ax.plot(px_2, py_2 ,pz_2, c = "red")
            ax.set_xlabel("x-Axis")
            ax.set_ylabel("y-Axis")
            ax.set_zlabel("z-Axis")
            ax.set_xlim3d(float(datamean[0]) - scale, float(datamean[0]) + scale)
            ax.set_ylim3d(float(datamean[1]) - scale, float(datamean[1]) + scale)
            ax.set_zlim3d(float(datamean[2]) - scale, float(datamean[2]) + scale)
            ax.legend(fontsize = "6")
            plt.show()
            yes = ['yes','y', 'ye']
            no = ['no','n']
            user_input = input('Do you want to continue? \nyes/y = continue \nno/n = stop\n')
            if user_input.lower() in yes:
                print("continuing")
                pass
            elif user_input.lower() in no:
                sys.exit("Program was stopped")
            else:
                sys.stdout.write("Please respond with 'yes' or 'no'")

    al,bl,linedistance = closestDistanceBetweenLines(p1,p2,p1_2,p2_2)    

    return axis1, axis2, datamean, scale, linedistance
    