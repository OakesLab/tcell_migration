import glob as glob                                            # grabbing file names
import pandas as pd                                            # making dataframe for exporting parameters
import numpy as np                                             # basic math
import skimage.io as io                                        # reading in images
import os
import trackpy as tp             # particle tracking toolbox
import matplotlib.pyplot as plt  # for plotting everything
from matplotlib import cm, colors
import seaborn as sns
import matplotlib.patches as mpatches
# from skimage.feature import peak_local_max
# import warnings
# warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)
# from image_plotting_tools import *
# from interactive_plotting_tools import *
# import czifile
import shutil
# from scipy import optimize               # for curve fitting

def  check_peak_locations(imstack, frame_num = 0, feature_size = 11, minmass = 2500, separation = 3, im_min_inten = None, im_max_inten = None):
    # find the peaks in the given frame 
    cells = tp.locate(imstack[frame_num], feature_size, minmass=minmass, separation = separation)

    # plot the images with the peaks overlaid
    plt.figure()
    plt.imshow(imstack[0], vmin = im_min_inten, vmax = im_max_inten)
    plt.plot(cells['x'], cells['y'], 'rx')
    plt.title('trackpy intensity features')
    plt.axis('off')

    # show the image
    plt.show()
    
    return

def calculate_track_parameters(cell_tracks_filtered, um_per_pixel = 1, frame_duration = 1):

    # set particle as the index
    cell_tracks_filtered_indexed = cell_tracks_filtered.set_index('particle')
    
    # make a list of all the unique particle values
    unique_particle=np.unique(cell_tracks_filtered['particle'].values)

    # empty lists to hold data
    trajectory_x, trajectory_y = [],[]
    displacement, pathlength = [],[]
    path_duration, path_duration_frames = [],[]
    average_velocity, effective_velocity = [],[]
    frame_list, first_frame_list = [],[]

    # loop through each particle id
    for puncta in unique_particle:
        # make a table of the points for each individual myosin puncta
        individual_table = cell_tracks_filtered_indexed.loc[puncta]
        trajectory_x.append(individual_table['x'].values)
    #     trajectory_xs.append(np.convolve(individual_table['x'], kernel, mode='valid'))
        trajectory_y.append(individual_table['y'].values)
    #     trajectory_ys.append(np.convolve(individual_table['y'], kernel, mode='valid'))
        frame_list.append(individual_table['frame'].values)
        first_frame_list.append(frame_list[-1][0])
    #     frame_list_s.append(frame_list[-1][int(np.floor(smoothing_window_size/2)):-int(np.ceil(smoothing_window_size/2))+1])
        path_duration.append(np.sum(np.diff(frame_list[-1])) * frame_duration)
        path_duration_frames.append(np.sum(np.diff(frame_list[-1])))
    #     path_duration_s.append(np.sum(np.diff(frame_list_s[-1]) * frame_duration))
        pathlength.append(np.sum(np.sqrt((np.diff(trajectory_x[-1])**2 + np.diff(trajectory_y[-1])**2))) * um_per_pixel)
    #     pathlength_s.append(np.sum(np.sqrt((np.diff(trajectory_xs[-1])**2 + np.diff(trajectory_ys[-1])**2))) * um_per_pixel)
        average_velocity.append(pathlength[-1] / path_duration[-1])
        displacement.append(np.sqrt((trajectory_x[-1][-1] - trajectory_x[-1][0])**2 + (trajectory_y[-1][-1] - trajectory_y[-1][0])**2) * um_per_pixel)
    #     displacement_s.append(np.sqrt((trajectory_xs[-1][-1] - trajectory_xs[-1][0])**2 + (trajectory_ys[-1][-1] - trajectory_ys[-1][0])**2) * um_per_pixel)
        effective_velocity.append(displacement[-1] / path_duration[-1])

    # make the dataframe
    cell_trackdata_df = pd.DataFrame()
    cell_trackdata_df['x'] = trajectory_x
    cell_trackdata_df['y'] = trajectory_y
    cell_trackdata_df['frames'] = frame_list
    cell_trackdata_df['displacement'] = displacement
    cell_trackdata_df['duration'] = path_duration
    cell_trackdata_df['pathlength'] = pathlength
    # cell_trackdata_df['x_smoothed'] = trajectory_xs
    # cell_trackdata_df['y_smoothed'] = trajectory_ys
    # cell_trackdata_df['displacement_smoothed'] = displacement_s
    # cell_trackdata_df['frames_smoothed'] = frame_list_s
    # cell_trackdata_df['duration_smoothed'] = path_duration_s
    # cell_trackdata_df['pathlength_smoothed'] = path_length_s
    cell_trackdata_df['average_velocity'] = average_velocity
    cell_trackdata_df['effective_velocity'] = effective_velocity
    cell_trackdata_df['first_frame'] = first_frame_list
    cell_trackdata_df['duration_frames'] = path_duration_frames

    return cell_trackdata_df