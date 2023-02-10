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
    plt.imshow(imstack[frame_num], vmin = im_min_inten, vmax = im_max_inten)
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

def plot_track_overlays(imstack, cell_trackdata_df, color_hue=None):
    # get the max projection of the image stack
    stack_sum = np.max(imstack, axis=0)
    # make a colormap of the right length
    if color_hue:
        cmap = plt.cm.turbo
        max_colormap = np.max(cell_trackdata_df[color_hue])
        norm = colors.Normalize(vmin=0, vmax=max_colormap)
    
    # make the figure
    fig, ax = plt.subplots()
    # plot the summed image so overlays are easy to see
    ax.imshow(stack_sum, cmap = 'Greys_r', vmin = np.min(stack_sum), vmax = .2*np.max(stack_sum))
    for index, row in cell_trackdata_df.iterrows():
        if color_hue:
            ax.plot(row['x'], row['y'], color = cmap(norm(row[color_hue])))
        else:
            ax.plot(row['x'], row['y'], color = 'r')
    ax.axis('off')
    if color_hue:
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax, label=color_hue)
    fig.show
    
    return

# need following variables as part of the function: filename, imstack
def make_movie_with_overlays(filename, imstack, cell_trackdata_df, im_min_inten = None, im_max_inten = None, color_hue = None):
    
    # make a colormap of the right length
    if color_hue:
        cmap = plt.cm.turbo
        max_colormap = np.max(cell_trackdata_df[color_hue])
        norm = colors.Normalize(vmin=0, vmax=max_colormap)
    
    # make a directory to store all the tracked images
    if filename.rfind('/') == -1:
        movie_folder = 'movie'
    else:
        movie_folder = filename[:filename.rfind('/')] + '/movie'
    # if the folder doesn't exist, make it 
    if os.path.isdir(movie_folder) == False:
        os.mkdir(movie_folder)
    
    # get the number of frames in the movie
    N_images = imstack.shape[0]
    # make the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.show()
    for frame in np.arange(0,N_images):
        # clear the axes 
        ax.clear()
        # show the image
        ax.imshow(imstack[frame], vmin=im_min_inten, vmax=im_max_inten, cmap='Greys')
        # make a reduced dataframe that has all tracks that start before that frame
        reduced_df = cell_trackdata_df[cell_trackdata_df['first_frame'] < frame]
        # loop through tracks
        for index, row in reduced_df.iterrows():
            # keep only those points that are in the previous frames
            x = row['x'][row['frames'] < frame]
            y = row['y'][row['frames'] < frame]
            if color_hue:
                ax.plot(x, y, color = cmap(norm(row[color_hue])))
            else:
                ax.plot(x, y, color = 'r')
        # turn the axes off
        ax.axis('off')
        # save the figure
        fig.savefig(movie_folder + '/cell_tracks_frame_%03d.png' % frame, dpi = 150, bbox_inches='tight')

    # get a list of the files in the folder
    file_list = sorted(glob.glob(movie_folder + '/cell_tracks_frame*.png'))
    # read in the first frame
    first_frame = io.imread(movie_folder + '/cell_tracks_frame_000.png')
    # keep just the first channel
    first_frame = first_frame[:,:,0]
    # make empty lists to find the rows,columns that have data
    rows, cols = [],[]
    for i in np.arange(0,first_frame.shape[0]):
        rows.append(len(np.unique(first_frame[i,:])))
    for j in np.arange(0,first_frame.shape[1]):
        cols.append(len(np.unique(first_frame[:,j])))

    # figure out which rows and columns have data
    rows_withdata = np.where(np.array(rows) > 1)
    cols_withdata = np.where(np.array(cols) > 1)

    row_begin = rows_withdata[0][0]
    row_end = rows_withdata[0][-1] + 1
    col_begin = cols_withdata[0][0]
    col_end = cols_withdata[0][-1] + 1

    # make an image stack without the padding
    plot_stack = np.zeros((N_images, row_end - row_begin, col_end - col_begin,3), dtype = 'uint8')
    # crop each image and add it to the stack
    for i,im_name in enumerate(file_list):
        im = io.imread(im_name)
        plot_stack[i] = im[row_begin:row_end,col_begin:col_end,0:3]

    # save the movie as a .tif stack
    io.imsave(filename[:-4] + '_movie.tif', plot_stack.astype('uint8'))  
    #remove the individual images and the folder
    shutil.rmtree(movie_folder)
    
    return

