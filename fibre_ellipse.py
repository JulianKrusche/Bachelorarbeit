import matplotlib.pyplot as plt
import matplotlib.patches as pltp
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from tkinter import filedialog as fd
import tkinter
import sys
import csv
import numpy as np
from tabulate import tabulate
from PIL import Image
import time

from elipse_main_functions_big import ellipse_along_axis
from elipse_main_functions import export_points_txt
from elipse_main_functions import export_points_txt_append
from elipse_main_functions import read_data
from elipse_main_functions import reduce_data
from elipse_main_functions import probe_top_edge
from elipse_main_functions import fibre_top_edge
from elipse_main_functions_big import check_axis

print("1. Select file of point data (.csv or .ply) \n2. Select file(s) of slice of the whole probe (.tiff or .png) \n3. Select file of slice of just the fibre (.tiff or .png)")
tkinter.Tk().withdraw()
filename = fd.askopenfilename()
filename_image1 = fd.askopenfilenames()
filename_image2 = fd.askopenfilename()
start = time.time()

coord_data, ply = read_data(filename)
#axis1, axis2, datamean, scale = check_axis(coord_data)

yes = ['yes','y', 'ye']
no = ['no','n']
if len(coord_data) > 50000000:
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

axis1, axis2, datamean, scale, linedistace = check_axis(coord_data)
print("fibre 1:", axis1)
pfp_x1, pfp_y1, pfp_z1, Xc1, Yc1, Zc1, height1, width1, lenght1 = ellipse_along_axis(coord_data, axis1, set_number_of_layers = 32, ply = ply)
print("fibre 2:",axis2)
pfp_x2, pfp_y2, pfp_z2, Xc2, Yc2, Zc2, height2, width2, lenght2 = ellipse_along_axis(coord_data, axis2, set_number_of_layers = 32, ply = ply)

imarray, probe_top_mean, probe_lower_mean = probe_top_edge(filename_image1)
im = Image.fromarray(imarray)
fibre_top = fibre_top_edge(filename_image2)
probe_height = probe_lower_mean - probe_top_mean
h01 = abs(fibre_top - probe_top_mean)

table = [["Output", "fibre 1", "fibre 2"], ["fibre height [mm]", round(height1 * 9.4 / 1000, 4), round(height2 * 9.4 / 1000, 4)], ["fibre width[mm]", round(width1 * 9.4 / 1000, 4), round(width2 * 9.4 / 1000, 4)], ["fibre lenght [mm]", round(lenght1 * 9.4 / 1000, 4), round(lenght2 * 9.4 / 1000, 4)], ["C0 [mm]", round(h01 * 9.4 / 1000, 4),""], ["probe height [mm]", round(probe_height * 9.4 / 1000, 4), ""], ["distance fibre lines [mm]", round(linedistace * 9.4 / 1000, 4), ""]]
print(tabulate(table, headers = "firstrow", tablefmt = "fancy_grid"))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

rwthblue50 = (0/255,84/255,159/255)
rwthred = (204/255,7/255,30/255)

ax.scatter(pfp_x1, pfp_y1, pfp_z1, color = rwthred, s = 0.1)
ax.plot_surface(Xc1, Yc1, Zc1, alpha = 0.4, color = rwthblue50)

ax.scatter(pfp_x2, pfp_y2, pfp_z2, color = "r", s = 0.1)
ax.plot_surface(Xc2, Yc2, Zc2, alpha = 0.4, color = rwthblue50)

ax.set_xlabel("x-Axis")
ax.set_ylabel("y-Axis")
ax.set_zlabel("z-Axis")
ax.set_xlim3d(float(datamean[0]) - scale * 1.1, float(datamean[0]) + scale * 1.1)
ax.set_ylim3d(float(datamean[1]) - scale * 1.1, float(datamean[1]) + scale * 1.1)
ax.set_zlim3d(float(datamean[2]) - scale * 1.1, float(datamean[2]) + scale * 1.1)
fake2Dline = mpl.lines.Line2D([0],[0], linestyle = "none", c = "blue", marker = 'o')
fake2Dline2 = mpl.lines.Line2D([0],[0], linestyle = "none", c = "red", marker = 'o')
ax.legend([fake2Dline, fake2Dline2], ["This is how the fibre should look like in GrassHopper \ndata is in the terminal", "These points are the edge points of the fibre \nfrom which the heigth and width is caculated"], numpoints = 1, fontsize= "9")

end = time.time()
print("Time:", round(end - start, 2), "seconds")

plt.show()
im.show()

user_input2 = input('Export data? \ntxt = create txt with details \ncsv = create csv with details \nno/n = end program \n')
if user_input2.lower() == "txt":
    with open('table.txt', 'w') as f:
        f.write(tabulate(table))
elif user_input2.lower() == "csv":
    with open('output_test.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(["Output", "Big fibre", "Small fibre"])
            writer.writerow(["fibre height [mm]", round(height1 * 9.4 / 1000, 5), round(height2 * 9.4 / 1000, 5)])
            writer.writerow(["fibre width[mm]", round(width1 * 9.4 / 1000, 5), round(width2 * 9.4 / 1000, 5)])
            writer.writerow(["fibre lenght [mm]", round(lenght1 * 9.4 / 1000, 5), round(lenght2 * 9.4 / 1000, 5)])
            writer.writerow(["C0 [mm]", round(h01 * 9.4 / 1000, 5), "",])
            writer.writerow(["probe height [Pixel] [mm?]", probe_height, round(probe_height * 9.4 / 1000, 5)])
elif user_input2.lower() in no:
        sys.exit("Program finished")
else:
    sys.stdout.write("Please respond with 'txt' or 'csv' or 'no'")


#export_points_txt(Xc1, Yc1, Zc1, "presentation1")
#export_points_txt_append(Xc2, Yc2, Zc2, "presentation1")

