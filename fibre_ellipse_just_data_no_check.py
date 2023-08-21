import matplotlib.pyplot as plt
import matplotlib.patches as pltp
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from tkinter import filedialog as fd
import tkinter
import sys
import numpy as np
from tabulate import tabulate
from PIL import Image


from elipse_main_functions_big import ellipse_along_axis
from elipse_main_functions import export_points_txt
from elipse_main_functions import export_points_txt_append
from elipse_main_functions import read_data
from elipse_main_functions import reduce_data
from elipse_main_functions import probe_top_edge
from elipse_main_functions import fibre_top_edge
from elipse_main_functions_big import check_axis

#print("1. Select file of point data (.csv or .ply)")
tkinter.Tk().withdraw()
filename = fd.askopenfilename()

#filename = "Z:/30_Transfer/Studenten/Julian.Krusche/01_Bachelorarbeit/Python/Rohdaten_overview/no_rotation.ply"

axis1 = "x"
axis2 = "z"

coord_data, ply = read_data(filename)
print("Number of points:", len(coord_data))

if len(coord_data) > 50000000:
    yes = ['yes','y', 'ye']
    no = ['no','n']
    reduce = ['reduce','r']
    user_input = input('Data contains more then 50 million points. Do you want to continue (program could crash) or do you want to to reduce the data to 20 Million Points? \nyes/y = continue \nno/n = stop \nreduce/r = reduce data \n')
    if user_input.lower() in yes:
        print("continuing")
        pass
    elif user_input.lower() in no:
        sys.exit("Program was stopped")
    elif user_input.lower() in reduce:
        print("Amount of points before reduction:", len(coord_data))
        coord_data = reduce_data(coord_data)
        print("Amount of points after reduction:", len(coord_data))
    else:
        sys.stdout.write("Please respond with 'yes' or 'no'")

np_coord_data = np.array(coord_data)
datamean = np_coord_data.mean(axis=0)
scale = float(np.max([[datamean[0] - np.min(coord_data[0])], [datamean[1] - np.min(coord_data[1])], [datamean[2] - np.min(coord_data[2])], [np.max(coord_data[0]) - datamean[0]], [np.max(coord_data[1]) - datamean[1]], [np.max(coord_data[2]) - datamean[2]]]))
    

print("fibre 1:", axis1)
pfp_x1, pfp_y1, pfp_z1, Xc1, Yc1, Zc1, height1, width1, lenght1 = ellipse_along_axis(coord_data, axis1, set_number_of_layers = 32, ply = ply)
print("fibre 2:",axis2)
pfp_x2, pfp_y2, pfp_z2, Xc2, Yc2, Zc2, height2, width2, lenght2 = ellipse_along_axis(coord_data, axis2, set_number_of_layers = 32, ply = ply)

table = [["Output", "fibre 1", "fibre 2"], ["fibre height [mm]", round(height1 * 9.4 / 1000, 5), round(height2 * 9.4 / 1000, 5)], ["fibre width[mm]", round(width1 * 9.4 / 1000, 5), round(width2 * 9.4 / 1000, 5)], ["fibre lenght [mm]", round(lenght1 * 9.4 / 1000, 5), round(lenght2 * 9.4 / 1000, 5)]]
print(tabulate(table, headers = "firstrow", tablefmt = "fancy_grid"))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

rwthblue = (0/255,84/255,159/255)
rwthbue50 = (142/255,186/255,229/255)
rwthred = (204/255,7/255,30/255)
major_ticks = np.arange(-500, 2500, 500)
minor_ticks = np.arange(-1000, 2000, 500)

ax.scatter(pfp_x1, pfp_y1, pfp_z1, color = rwthred, s = 0.1)
ax.plot_surface(Xc1, Yc1, Zc1, alpha = 0.4, color = rwthblue)

ax.scatter(pfp_x2, pfp_y2, pfp_z2, color = rwthred, s = 0.1)
ax.plot_surface(Xc2, Yc2, Zc2, alpha = 0.4, color = rwthblue)

ax.set_xlabel("x-Axis")
ax.set_ylabel("y-Axis")
ax.set_zlabel("z-Axis")
ax.set_xticks(major_ticks)
ax.set_yticks(minor_ticks)
ax.set_zticks(minor_ticks)
ax.set_xlim3d(float(datamean[0]) - scale * 1.1, float(datamean[0]) + scale * 1.1)
ax.set_ylim3d(float(datamean[1]) - scale * 1.1, float(datamean[1]) + scale * 1.1)
ax.set_zlim3d(float(datamean[2]) - scale * 1.1, float(datamean[2]) + scale * 1.1)
fake2Dline = mpl.lines.Line2D([0],[0], linestyle = "none", c = rwthblue, marker = 'o')
fake2Dline2 = mpl.lines.Line2D([0],[0], linestyle = "none", c = rwthred, marker = 'o')
ax.legend([fake2Dline, fake2Dline2], ["This is how the fibre should \nlook like in GrassHopper", "edge points of layers"], numpoints = 1, fontsize= "10")
plt.show()

"""
with open('table.txt', 'w') as f:
f.write(tabulate(table))
"""