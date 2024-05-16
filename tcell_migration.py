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
import subprocess                                  # for executing commands from the terminal
from skimage.registration import phase_cross_correlation       # image registration
from scipy.fftpack import fft2, ifft2, ifftshift               # FFT for image registration
from scipy.ndimage.morphology import distance_transform_edt   # for distance mask transform

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

def calculate_track_parameters(cell_tracks_filtered, filename='.tif', um_per_pixel = 1, frame_duration = 1):

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
    velocity = []

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
        v = np.sqrt(np.diff(individual_table['x'].values)**2 + np.diff(individual_table['y'].values)**2) * um_per_pixel / frame_duration
        velocity.append(np.insert(v, 0, 0))

    # make the dataframe
    cell_trackdata_df = pd.DataFrame()
    cell_trackdata_df['x'] = trajectory_x
    cell_trackdata_df['y'] = trajectory_y
    cell_trackdata_df['velocity'] = velocity
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

    cell_trackdata_df.to_hdf(filename[:-4] + '_trackdata.h5', key='tracks', mode='w')

    # make a reduced dataframe withoutx,y,frameslist
    cell_trackdata_df_csv = cell_trackdata_df.drop(['x','y','frames'], axis=1)
    cell_trackdata_df_csv.to_csv(filename[:-4] + '_trackdata.csv')

    return cell_trackdata_df

def plot_track_overlays(imstack, cell_trackdata_df, filename='.tif', color_hue = None, max_colormap = None, save_plot = True):
    # get the max projection of the image stack
    stack_sum = np.max(imstack, axis=0)
    # make a colormap of the right length
    if color_hue:
        cmap = plt.cm.rainbow
        if max_colormap is None:
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
            ax.plot(row['x'], row['y'])
    ax.axis('off')
    if color_hue:
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax, label=color_hue)
    fig.show

    if save_plot:
        if color_hue:
            fig.savefig(filename[:-4] + '_' + color_hue + '_trackoverlay.png', dpi=300, bbox_inches='tight')
        else:
            fig.savefig(filename[:-4] + '_trackoverlay.png', dpi = 300, bbox_inches='tight')
    
    return

# need following variables as part of the function: filename, imstack
def make_movie_with_overlays(filename, imstack, cell_trackdata_df, im_min_inten = None, im_max_inten = None, color_hue = None, max_colormap = None, savename = None):
    
    # make the filename for saving
    if savename is None:
        savename = filename[:-4] + 'overlays.tif'
    else:
        savename = filename[:-4] + '_' + savename + '_overlays.tif'

    # make a colormap of the right length
    if color_hue:
        cmap = plt.cm.rainbow
        if max_colormap is None:
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
        fig.savefig(movie_folder + '/cell_tracks_frame_%03d.png' % frame, dpi = 150, bbox_inches='tight', pad_inches = 0)

    # get a list of the files in the folder
    file_list = sorted(glob.glob(movie_folder + '/cell_tracks_frame*.png'))
    # read in the first frame
    first_frame = io.imread(movie_folder + '/cell_tracks_frame_000.png')
    # keep just the first channel
    first_frame = first_frame[:,:,0]
    # get the shape of the images
    N_rows, N_cols = first_frame.shape

    # make a matrix to hold everything
    plot_stack = np.zeros((N_images, N_rows, N_cols,3), dtype = 'uint8')
    # Add each image to the stack
    for i,im_name in enumerate(file_list):
        im = io.imread(im_name)
        plot_stack[i] = im[:N_rows,:N_cols,0:3]

    # save the movie as a .tif stack
    # io.imsave(filename[:-4] + '_movie.tif', plot_stack.astype('uint8'))
    io.imsave(savename, plot_stack.astype('uint8'))   

    # save_timelapse_as_movie(plot_stack, filename[:-4] + 'overlays.tif')
    save_timelapse_as_movie(plot_stack, savename)

    #remove the individual images and the folder
    shutil.rmtree(movie_folder, onerror=ignore_extended_attributes)
    
    return

def ignore_extended_attributes(func, filename, exc_info):
    is_meta_file = os.path.basename(filename).startswith("._")
    if not (func is os.unlink and is_meta_file):
        raise

def plot_roseplot(cell_trackdata_df, filename='.tif', um_per_pixel = 1, color_hue = None, max_colormap = None, xlimits = (None,None), ylimits = (None,None), save_plot=True):
    
    # set the limits of the plot if a single number is passed
    if (type(ylimits) is int) | (type(ylimits) is float):
        ylimits = (-1*ylimits, ylimits)
    if (type(xlimits) is int) | (type(xlimits) is float):
        xlimits = (-1*xlimits, xlimits)

    # if a color hue is given, sort the data and make the colormap for that range
    if color_hue:
        cell_trackdata_df = cell_trackdata_df.sort_values(by = color_hue, ascending=False)
        cmap = plt.cm.rainbow
        if max_colormap is None:
            max_colormap = np.max(cell_trackdata_df[color_hue])
        norm = colors.Normalize(vmin=0, vmax=max_colormap)

    # make the plot
    fig, ax = plt.subplots()
    # loop through the data and assign colors
    for index, row in cell_trackdata_df.iterrows():
        if color_hue:
            ax.plot((row['x'] - row['x'][0]) * um_per_pixel, (row['y'] - row['y'][0]) * um_per_pixel, color = cmap(norm(row[color_hue])))
        else:
            ax.plot((row['x'] - row['x'][0]) * um_per_pixel, (row['y'] - row['y'][0]) * um_per_pixel)

    # make the plot square and set the limits
    ax.axis('square')
    ax.set_ylim(ylimits)
    ax.set_xlim(xlimits)
    # add a colorbar if there's a hue
    if color_hue:
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax, label=color_hue)
    # show the image
    fig.show

    # save the image
    if save_plot:
        fig.savefig(filename[:-4] + '_roseplot.eps', format = 'eps', bbox_inches='tight')
        fig.savefig(filename[:-4] + '_roseplot.png', format = 'png', dpi=300, bbox_inches='tight')
    
    return

def save_timelapse_as_movie(imstack, filename='.tif', framerate = 15, warning_flag=False):
    '''Save a timelapse as a movie using the **kwargs to make a ffmpeg command to run in the terminal'''
    
    # make a temp folder to hold the image series
    if os.path.isdir('temp_folder') == False:
        os.mkdir('temp_folder')
    
    # check stack shape to make sure it's even or ffmpeg will throw an error
    if imstack.shape[1] % 2:
        imstack = imstack[:,:-1,:]
    if imstack.shape[2] % 2:
        imstack = imstack[:,:,:-1]

    # determine the number of images
    N_images = imstack.shape[0]
    
    # write image series
    # if a timestamp is included
    for plane in np.arange(0,N_images):
        io.imsave('temp_folder/movie%04d.tif' % plane, imstack[plane], check_contrast=False)

    # generate the ffmpeg command
    movie_str = 'ffmpeg -y -f image2 -r ' + str(framerate) + ' -i temp_folder/movie%04d.tif -vcodec libx264 -crf 25 -pix_fmt yuv420p ' + filename[:-4] + '_movie.mp4'
    
    # run ffmpeg
    if warning_flag:
        subprocess.call(movie_str, shell=True)
    else:
        subprocess.call(movie_str, shell=True, stderr = subprocess.STDOUT, stdout = subprocess.DEVNULL)
    
    # delete the temp folder and files
    shutil.rmtree('temp_folder', onerror=ignore_extended_attributes)
    
    return

def make_movie_of_raw_data(imstack, filename, min_inten = None, max_inten = None):
    
    # make a copy of the stack
    imstack_8bit = imstack.copy()
    # set the min and max intensities if they're note defined
    if max_inten is None:
        max_inten = np.max(imstack_8bit) * 0.2
    if min_inten is None:
        min_inten = np.min(imstack_8bit)
    # set the max intensity
    imstack_8bit[imstack_8bit > max_inten] = max_inten
    # set the min intensity
    imstack_8bit[imstack_8bit < min_inten] = min_inten
    # normalize to an 8bit image
    imstack_8bit = imstack_8bit - min_inten
    imstack_8bit = imstack_8bit / np.max(imstack_8bit)  * 255
    # save the movie
    save_timelapse_as_movie(imstack_8bit.astype('uint8'), filename)
    
    return

def register_stack_firstframe(imagename, save_result = True):
    # check if it's a string or a matrix and read in the image
    if isinstance(imagename, str):
        image_stack = io.imread(imagename)
        imagename = imagename[:-4]
    else:
        image_stack = imagename
        imagename = 'imagestack'
        
    # Get the number of images in the stack
    N_images, N_rows, N_cols = image_stack.shape

    # Create an empty array to hold the registered images
    image_stack_registered = np.zeros_like(image_stack)

    # Create an empty array to hold the shift coordinates to reference for use in other channels
    shift_coordinates = np.zeros((N_images, 4))
    shift_coordinates2 = np.zeros((N_images, 4))
    # Create a matrix with the row and column numbers for the registered image calculation
    Nr = ifftshift(np.arange(-1 * np.fix(N_rows/2), np.ceil(N_rows/2)))
    Nc = ifftshift(np.arange(-1 * np.fix(N_cols/2), np.ceil(N_cols/2)))
    Nc, Nr = np.meshgrid(Nc, Nr)

    # Define the subpixel resolution
    subpixel_resolution = 10
    
    # Define reference frame
    reference_image = image_stack[0]
    
    # For loop to register each plane in the stack
    for plane in np.arange(1,N_images):
        # Read in the image you want to register
        next_image = image_stack[plane,:,:]
        
        # Perform the subpixel registration
        shift, error, diffphase = phase_cross_correlation(reference_image, next_image, upsample_factor=subpixel_resolution)
        # Store the shift coordinates
        shift_coordinates[plane] = np.array([shift[0] , shift[1] , error, diffphase])

        # Calculate the shifted image
        shifted_image_fft = fft2(next_image) * np.exp(
                1j * 2 * np.pi * (-shift_coordinates[plane,0] * Nr / N_rows - shift_coordinates[plane,1] * Nc / N_cols))
        shifted_image_fft = shifted_image_fft * np.exp(1j * diffphase)
        shifted_image = np.abs(ifft2(shifted_image_fft))
        image_stack_registered[plane,:,:] = shifted_image.copy()

    # set the first plane to the same as the stack
    image_stack_registered[0] = image_stack[0]
    
    if save_result:
        io.imsave(imagename + '_registered.tif', image_stack_registered.astype('uint16'), check_contrast = False)
    
    return image_stack_registered, shift_coordinates

def shift_image_stack(image_stack_name, shift_coordinates):
    """
    Register an image stack based on a previously registered stack of images
    
    Parameters
    ----------
    image_stack_name  :  str         - name of the image stack to be registered
    shift_coordinates :  4xN ndarray - contains N rows of [ row shift, col shift, error, diffphase] 
                                       from skimage.feature.register_translation output. N must be the same number
                                       of images in the stack
    
    Output
    ------
    Saves the registered stack of images with the same name with '_registered' appended
    """
    
    # read in image stack
    image_stack = io.imread(image_stack_name)
    
    # correct the stack shape if there's only one image
    if len(image_stack.shape) == 2:
        temp = np.zeros((1,image_stack.shape[0],image_stack.shape[1]))
        temp[0] = image_stack.copy()
        image_stack = temp.copy()

    # Get the shape of your stack
    N_planes, N_rows, N_cols = image_stack.shape
    
    # Create a matrix with the row and column numbers for the registered image calculation
    Nr = ifftshift(np.arange(-1 * np.fix(N_rows/2), np.ceil(N_rows/2)))
    Nc = ifftshift(np.arange(-1 * np.fix(N_cols/2), np.ceil(N_cols/2)))
    Nc, Nr = np.meshgrid(Nc, Nr)
    
    # Create an empty array to hold the registered image
    image_registered = np.zeros((N_planes, N_rows, N_cols))
    
    # register each plane based on the provided coordinates
    for plane in np.arange(0,N_planes):
        raw_image = image_stack[plane]
        shifted_image_fft = fft2(raw_image) * np.exp(
        1j * 2 * np.pi * (-shift_coordinates[plane,0] * Nr / N_rows - shift_coordinates[plane,1] * Nc / N_cols))
        shifted_image_fft = shifted_image_fft * np.exp(1j * shift_coordinates[plane,3])
        shifted_image = np.abs(ifft2(shifted_image_fft))
        image_registered[plane] = shifted_image.copy()
    
    # new file name
    image_registered_name = image_stack_name[:-4] + '_registered.tif'
    
    # Save the registered stack
    io.imsave(image_registered_name, image_registered.astype('uint16'), check_contrast=False)
    
    return

def calculate_micropattern_velocity(cell_trackdata_df, pattern_mask, filename = '.tif'):
    # lists to hold new data
    fn_on_velocity, fn_off_velocity = [],[]
    pattern_pts, pattern_crossing = [], []

    # read in micropattern if given a string
    if isinstance(pattern_mask, str):
        pattern_mask = io.imread(pattern_mask).astype('bool')

    # make the distance mask of the pattern
    distance_mask = distance_transform_edt(np.invert(pattern_mask))



    for index, row in cell_trackdata_df.iterrows():
        # get track data
        x = row['x']
        y = row['y']
        v = row['velocity']
        # remove first point with velocity 0
        x = x[1:]
        y = y[1:]
        v = v[1:]

        # check which points are in the pattern
        in_mask = []
        for (xpt,ypt) in zip(x,y):
            # have to switch x and y because of the difference for arrays versus plotting
            # if pattern_mask[int(ypt),int(xpt)]:
            # use the distance mask because our threshold isn't perfect so we include points that are within 5 pixels
            if distance_mask[int(ypt),int(xpt)] < 5:
                in_mask.append(True)
            else:
                in_mask.append(False)

        # calculate the average velocity for each condition
        if np.sum(in_mask) > 0:
            fn_on_velocity.append(np.mean(v[in_mask]))
            if np.sum(in_mask) == len(in_mask):
                fn_off_velocity.append(np.nan)
            else:
                fn_off_velocity.append(np.mean(v[np.invert(in_mask)]))
        else:
            fn_on_velocity.append(np.nan)
            fn_off_velocity.append(np.mean(v[np.invert(in_mask)]))
 
        if np.isnan(fn_on_velocity[-1]) or np.isnan(fn_off_velocity[-1]):
            pattern_crossing.append(False)
        else:
            pattern_crossing.append(True)
        
        
        # save the list of on/off points
        pattern_pts.append(in_mask)

    # add things to the dataframe
    cell_trackdata_df['pattern_crossing'] = pattern_crossing
    cell_trackdata_df['fn_on_velocity'] = fn_on_velocity
    cell_trackdata_df['fn_off_velocity'] = fn_off_velocity
    cell_trackdata_df['pattern_pts'] = pattern_pts
    
    # save the dataframe
    cell_trackdata_df.to_hdf(filename[:-4] + '_pattern_trackdata.h5', key='tracks', mode='w')

    # make a dataframe for saving as csv with data
    cell_trackdata_df_csv = cell_trackdata_df.copy()
    cell_trackdata_df_csv = cell_trackdata_df_csv.drop(['x','y','velocity','frames','pattern_pts'], axis=1)
    cell_trackdata_df_csv.to_csv(filename[:-4] + '_pattern_trackdata.csv')
    
    return cell_trackdata_df

def plot_micropattern_comparison(cell_trackdata_df, filename='.tif', ylimits = (None,None), save_plot=True):
    # set the limits of the plot if a single number is passed
    if (type(ylimits) is int) | (type(ylimits) is float):
        ylimits = (-1*ylimits, ylimits)

    # reduce the dataframe to only those cells that cross the micropattern
    cell_trackdata_crossing_df = cell_trackdata_df[cell_trackdata_df['pattern_crossing'] == True]

    # make the figure
    crossing_fig, crossing_ax = plt.subplots()
    # plot the boxplot
    crossing_ax.boxplot([cell_trackdata_crossing_df['fn_on_velocity'],cell_trackdata_crossing_df['fn_off_velocity']], labels=['On','Off'])
    # plot the paired data
    for index, row in cell_trackdata_crossing_df.iterrows():
        crossing_ax.plot([1,2],[row['fn_on_velocity'], row['fn_off_velocity']], '.k-', alpha=0.2)
    crossing_ax.set_ylim(ylimits)
    crossing_ax.set_ylabel('Average Velocity')
    crossing_fig.show()

    # save the image
    if save_plot:
        crossing_fig.savefig(filename[:-4] + '_micropattern_comparison.eps', format = 'eps', bbox_inches='tight')
        crossing_fig.savefig(filename[:-4] + '_micropattern_comparison.png', format = 'png', dpi=300, bbox_inches='tight')
    
    return

def plot_instanteous_velocity_overlay(imstack, cell_trackdata_df, filename = '.tif', min_inten = None, max_inten = None, save_plot = True):
    color_hue='average_velocity'
    
    # set the min and max intensities if they're note defined
    if max_inten is None:
        max_inten = np.max(imstack) * 0.2
    if min_inten is None:
        min_inten = np.min(imstack)
    
    # make colormap
    cmap = plt.cm.rainbow
    max_colormap = 1.25*np.max(cell_trackdata_df[color_hue])
    norm = colors.Normalize(vmin=0, vmax=max_colormap)
    
    # plot the last frame of the image stack
    fig, ax = plt.subplots()
    ax.imshow(imstack[-1], vmin = min_inten, vmax = max_inten, cmap='Greys')
    # loop through tracks
    for index, row in cell_trackdata_df.iterrows():
        # keep only those points that are in the previous frames
        x = row['x']
        y = row['y']
        v = row['velocity']
        for (xpt, ypt, vpt) in zip(x,y,v):
            ax.scatter(xpt, ypt, s=2, facecolors='none', edgecolors=cmap(norm(vpt)))
    # turn off the axes
    ax.axis('off')
    # add a colorbar
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax, label=color_hue)
    fig.show()
    # save figure
    # save the image
    if save_plot:
        fig.savefig(filename[:-4] + '_instantaneous_velocity_overlay.eps', format = 'eps', bbox_inches='tight')
        fig.savefig(filename[:-4] + '_instantaneous_velocity_overlay.png', format = 'png', dpi=300, bbox_inches='tight')
    
    return

def update_overlap_df(track_overlap_df, trackA_id, trackA_x, trackA_y, trackA_frames, trackA_velocity,  trackB_id, trackB_x, trackB_y, trackB_frames, trackB_velocity, umperpixel=1.1):
    #empty lists to hold new data
    leader_id, follower_id = [],[]
    trackA_first_frame, trackB_first_frame = [],[]
    trackA_overlap_velocity, trackB_overlap_velocity = [],[]
    leader_velocity, follower_velocity = [],[]
    trackA_pathlength, trackB_pathlength = [],[]
    # calculate the path lengths of each track
    pathlengthA = np.sqrt((np.diff(trackA_x)**2 + np.diff(trackA_y)**2)) * umperpixel
    pathlengthA = np.insert(pathlengthA, 0, 0)
    pathlengthB = np.sqrt((np.diff(trackB_x)**2 + np.diff(trackB_y)**2)) * umperpixel
    pathlengthB = np.insert(pathlengthB, 0, 0)
    # loop through each labeled row
    for id,row in track_overlap_df.iterrows():
        # find the first frame of each track
        trackA_first_frame.append(trackA_frames[row['tA_start'].astype('int')])
        trackB_first_frame.append(trackB_frames[row['tB_start'].astype('int')])
        # find the velocity in the overlap region - the +1 comes from making sure to include the last point in the overlap region
        trackA_overlap_velocity.append(np.mean(trackA_velocity[row['tA_start'].astype('int'):row['tA_end'].astype('int')+1]))
        trackB_overlap_velocity.append(np.mean(trackB_velocity[row['tB_start'].astype('int'):row['tB_end'].astype('int')+1]))
        trackA_pathlength.append(np.sum(pathlengthA[row['tA_start'].astype('int'):row['tA_end'].astype('int')+1]))
        trackB_pathlength.append(np.sum(pathlengthB[row['tB_start'].astype('int'):row['tB_end'].astype('int')+1]))
        if trackA_first_frame[-1] < trackB_first_frame[-1]:
            leader_velocity.append(trackA_overlap_velocity[-1])
            follower_velocity.append(trackB_overlap_velocity[-1])
            leader_id.append(trackA_id)
            follower_id.append(trackB_id)
        else:
            leader_velocity.append(trackB_overlap_velocity[-1])
            follower_velocity.append(trackA_overlap_velocity[-1])
            leader_id.append(trackB_id)
            follower_id.append(trackA_id)
    track_overlap_df['tA_first_frame'] = trackA_first_frame
    track_overlap_df['tB_first_frame'] = trackB_first_frame
    track_overlap_df['tA_total'] = track_overlap_df['tA_end'] - track_overlap_df['tA_start']
    track_overlap_df['tB_total'] = track_overlap_df['tB_end'] - track_overlap_df['tB_start']
    track_overlap_df['tA_length'] = trackA_pathlength
    track_overlap_df['tB_length'] = trackB_pathlength
    track_overlap_df['tA_overlap_velocity'] = trackA_overlap_velocity
    track_overlap_df['tB_overlap_velocity'] = trackB_overlap_velocity
    track_overlap_df['leader_velocity'] = leader_velocity
    track_overlap_df['follower_velocity'] = follower_velocity
    track_overlap_df['leader_id'] = leader_id
    track_overlap_df['follower_id'] = follower_id
    track_overlap_df['velocity_difference'] = track_overlap_df['follower_velocity'] - track_overlap_df['leader_velocity']
    return track_overlap_df

def get_track_details(track_data_df, track_id):
    track_x = track_data_df.x[track_id]
    track_y = track_data_df.y[track_id]
    track_frames = track_data_df.frames[track_id]
    track_velocity = track_data_df.velocity[track_id]
    return track_x, track_y, track_frames, track_velocity

def make_distance_map(track0_x, track0_y, track1_x, track1_y, gap = 2):
    # make the track x y into a single array
    track0 = np.array([track0_x,track0_y]).T
    track1 = np.array([track1_x,track1_y]).T
    # calculate the euclidean distance between each point
    dist_map = cdist(track0,track1, metric='euclidean')
    # make the mask from that distance map
    track_distance_mask = dist_map < gap
    return track_distance_mask

def correct_track_distance_map(track_distance_mask, gap = 2, min_area = 3):
    # label the mask
    labeled_track_distance_mask = label(track_distance_mask)
    # get the label values
    unique_labels = np.unique(labeled_track_distance_mask)[1:]   # dropping zero since that's background
    # empty mask to hold output
    connecting_mask = np.zeros(track_distance_mask.shape)
    # loop through each label
    for lab in unique_labels:
        # make the mask for that label alone
        lab_mask = labeled_track_distance_mask == lab
        # get distance transform for that mask
        lab_distance = distance_transform_edt(np.invert(lab_mask))
        # only keep the pixels less than the gap
        lab_distance = lab_distance < gap
        # add only those pixels to the connecting mask
        connecting_mask += lab_distance
    # keep only those points that are connectin multiple labels
    connecting_mask = connecting_mask > 1
    # add those connecting pixels back to the track_distance_map
    track_distance_mask += connecting_mask
    # label the new track_distance_map
    track_mask_label = label(track_distance_mask)
    # get region props
    track_overlap_table = regionprops_table(track_mask_label, properties=['area','bbox','label'])
    # turn it into a dataframe
    track_overlap_df = pd.DataFrame(track_overlap_table)
    # get rid of areas smaller than the minimum
    track_overlap_df = track_overlap_df[track_overlap_df.area >= min_area]
    track_overlap_df.rename(columns={"bbox-0": "tA_start", "bbox-1": "tB_start", "bbox-2": "tA_end","bbox-3": "tB_end"}, inplace = True)
    return track_distance_mask, track_overlap_df

def get_condition(filename):
    filename = filename.lower()
    condition = 'unknown'
    if 'fibronectin' in filename:
        condition = 'fibronectin'
    if 'icam' in filename:
        condition = 'ICAM-1'
    if 'uncoated' in filename:
        condition = 'uncoated'
    if 'dmso' in filename:
        condition = 'dmso'
    if 'mmp' in filename:
        condition = 'mmp'
    return condition
