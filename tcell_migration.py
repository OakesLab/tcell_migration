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
def make_movie_with_overlays(filename, imstack, cell_trackdata_df, im_min_inten = None, im_max_inten = None, color_hue = None, max_colormap = None):
    
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
    io.imsave(filename[:-4] + '_movie.tif', plot_stack.astype('uint8'))  
    #remove the individual images and the folder
    shutil.rmtree(movie_folder)

    save_timelapse_as_movie(plot_stack, filename[:-4] + 'overlays.tif')
    
    return

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
    shutil.rmtree('temp_folder')
    
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
    io.imsave(image_registered_name, image_registered.astype('uint16'))
    
    return

