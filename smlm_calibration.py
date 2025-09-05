import skimage as skim 
import scipy as scp
from scipy.optimize import curve_fit
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile
from tkinter import filedialog
from tkinter import *

class smlm_calibration:

    def __init__(self, path, markers, ref_plane, dz, pixelsize):

        self.root_path = path 
        self.filepath = None
        self.filename = None
        self.markers_inherit = markers
        self.markers = {}
        self.biplane_markers = {}
        self.cal = {}
        self.output_path = None
        self.dz = dz # nm default
        self.P = {}
        self.FP = {} # focal planes 
        self.rr = 13
        self.ref_plane = ref_plane
        self.zrange=800
        self.gauss_sigma = 1.5 # sigma for DoG gaussian kernel
        self.pixelsize = pixelsize
        self.data_ext = ".tif"
        self.max_marker = 3


    def biplane_calibration(self):
        self.filepath, self.filename = self.get_filename()
        image = self.load_data()
        
        C, Z, X, Y = image.shape
        print(f"Loaded image {self.filename} of shape {image.shape}")
        
        # Identify focal plane per subimage
        for c in tqdm(range(C), "Finding focal plane"):
            self.FP[c] = find_focal_plane(image[c, ...])

        # Identify beads in focal plane
        for c in tqdm(range(C), "Finding markers"):
            self.markers[c] = self.locs_from_2d(image[c, self.FP[c], ...])

        # Match the localizations across planes
        for c in tqdm(range(C - 1), "Match markers"):
            ref, tar = self.match_markers(self.markers[c], self.markers[c + 1])
            self.biplane_markers[c] = {
                'ref': ref,
                'tar': tar,
                'ref_i': c,
                'tar_i': c + 1
            }

            # Visualize markers for the focal slice of the matched planes
            visualize_focal_markers(
                image[c, ...],  # Stack 1
                image[c + 1, ...],  # Stack 2
                ref,  # Markers for Stack 1
                tar,  # Markers for Stack 2
                (self.FP[c], self.FP[c + 1])  # Focal planes for Stack 1 and 2
            )

        # Perform biplane calibration
        #cal = {'slope': {}}
        cal = {'slope': []}
        params_ = []
        for c in tqdm(self.biplane_markers.keys(), "Calibrate biplane PSF widths"):
            ir, it = self.biplane_markers[c]['ref_i'], self.biplane_markers[c]['tar_i']
            mr, mt = self.biplane_markers[c]['ref'], self.biplane_markers[c]['tar']
            z0 = np.mean([self.FP[ir], self.FP[it]])
            slope, sigma1, sigma2 = self.biplane_slope(image[ir, ...], image[it, ...], mr, mt, z0)
            #cal['slope'][c] = slope
            cal['slope'].append(slope)
            if not params_: # append first layer
                params_.append(sigma1) 
            # always add last processed layer 
            params_.append(sigma2) 

         # (amplitude, mean, sigma, offset)
        params_array_ = np.array(params_)
        cal['amp'] = np.median(params_array_[:,0]) 
        cal['ddz'] = np.median(params_array_[:,1])  #np.mean([p for p in [params_.values()][1]]) 
        cal['sigma'] = np.median(params_array_[:,2]) 
        cal['offset'] = np.median(params_array_[:,3]) 
        return cal
    

    def biplane_slope(self, stack1, stack2, markers1, markers2, z0):
        """Main function to calculate the biplane slope."""
        num_slices, height, width = stack1.shape
        dz_abs = np.abs(self.dz)  # Ensure dz is positive
        dz = self.dz

        slice_multiple = round(self.zrange / dz_abs)
        slice_start = int(max(0, z0 - slice_multiple))
        slice_end = int(min(num_slices, z0 + slice_multiple))

        slice_multiple = round(self.zrange / dz_abs)
        slice_start = int(max(0, z0 - slice_multiple))
        slice_end = int(min(num_slices, z0 + slice_multiple))

        # Extract sigma_y values for each stack
        sigma_y_values_stack1 = self.extract_sigma_y_values(stack1, markers1, slice_start, slice_end, height, width)
        sigma_y_values_stack2 = self.extract_sigma_y_values(stack2, markers2, slice_start, slice_end, height, width)

        # Fit negative 1D Gaussians to the traces
        mean_params_1 = self.fit_gaussians_to_traces(sigma_y_values_stack1)
        mean_params_2 = self.fit_gaussians_to_traces(sigma_y_values_stack2)
        # (amplitude, mean, sigma, offset)
        #mean_params_1[2] = [self.dz*s for s in mean_params_1[2]]
        #mean_params_2[2] = [self.dz*s for s in mean_params_2[2]]

        mean_params_1 = np.mean(mean_params_1, axis=0)
        mean_params_2 = np.mean(mean_params_2, axis=0)

        #mean_sigma_stack2 = np.mean([self.dz*s for s in fit_gaussians_to_traces(sigma_y_values_stack2)])

        # Calculate the slope from the differences
        slope = calculate_slope(sigma_y_values_stack1, sigma_y_values_stack2, dz)

        return slope, mean_params_1, mean_params_2



    def fit_gaussians_to_traces(self, sigma_y_values):
        """Fit negative 1D Gaussians to traces and return mean sigma values."""
        mean_params = []
        for trace in sigma_y_values:
            x = np.arange(len(trace))
            y = np.array(trace)
            valid_idx = ~np.isnan(y)
            if valid_idx.sum() > 1:
                mean_param = fit_negative_1D_gaussian(x[valid_idx], y[valid_idx])
                mean_param[2] = mean_param[2]*self.dz # scale sigma in z to nm
                mean_params.append(mean_param)
            else:
                temp = np.empty((1,4))
                temp[:] = np.nan
                #mean_sigmas.append(np.nan)
                mean_params.append(temp)
        return mean_params
    

    def extract_sigma_y_values(self, stack, markers, slice_start, slice_end, height, width):
        """Optimized extraction of sigma_y values for markers within a range of slices."""
        sigma_y_values = []
        rr = self.rr
        max_marker = min(len(markers), self.max_marker)
        pixelsize = self.pixelsize

        for marker in markers[:max_marker]:
            x_center, y_center, _ = marker
            x_start = max(0, int(x_center) - rr)
            x_end = min(height, int(x_center) + rr)
            y_start = max(0, int(y_center) - rr)
            y_end = min(width, int(y_center) + rr)

            roi_height = x_end - x_start
            roi_width = y_end - y_start

            # Precompute meshgrid for the ROI shape
            X, Y = np.meshgrid(np.arange(roi_width), np.arange(roi_height))
            xy_data = np.vstack((X.ravel(), Y.ravel()))

            marker_sigma_y_values = []

            for i in range(slice_start, slice_end):
                roi = stack[i, x_start:x_end, y_start:y_end]
                z_data = roi.ravel()

                try:
                    max_val = np.max(roi)
                    min_val = np.min(roi)
                    initial_params = [max_val, roi_width / 2, roi_height / 2, 3, 3, min_val]
                    bounds = (
                        [min_val, 0.0, 0.0, 0.1, 0.1, 0],
                        [np.inf, roi_width, roi_height, np.inf, np.inf, max_val]
                    )

                    params, _ = curve_fit(gaussian_2D, xy_data, z_data, p0=initial_params, bounds=bounds)
                    sigma_y = params[4] * pixelsize
                    marker_sigma_y_values.append(sigma_y)
                except Exception:
                    marker_sigma_y_values.append(np.nan)

            sigma_y_values.append(marker_sigma_y_values)

        return sigma_y_values

    '''
    def extract_sigma_y_values(self, stack, markers, slice_start, slice_end, height, width):
        """Extract sigma_y values for markers within a range of slices."""
        sigma_y_values = []
        rr=self.rr

        for count, marker in enumerate(markers):
            if count >= self.max_marker:
                break

            marker_sigma_y_values = []

            for i in range(slice_start, slice_end):
                # Extract slice data
                slice_data = stack[i]
                x_center, y_center, _ = marker

                # Extract ROI
                roi = slice_data[
                    max(0, int(x_center) - rr):min(height, int(x_center) + rr),
                    max(0, int(y_center) - rr):min(width, int(y_center) + rr)
                ]

                # Generate fitting data
                X, Y = np.meshgrid(np.arange(roi.shape[1]), np.arange(roi.shape[0]))
                xy_data = np.vstack((X.ravel(), Y.ravel()))
                z_data = roi.ravel()

                try:
                    # Initial parameters for Gaussian fit
                    initial_params = [
                        np.max(roi), roi.shape[1] / 2, roi.shape[0] / 2, 3, 3, np.min(roi)
                    ]
                    bounds = ([np.min(roi), 0.0, 0.0, 0.1, 0.1, 0]
                              , [np.inf, roi.shape[1], roi.shape[0], np.inf, np.inf, np.max(roi)])
                    # Fit the 2D Gaussian
                    # amplitude, x0, y0, sigma_x, sigma_y, offset
                    params, _ = curve_fit(gaussian_2D, xy_data, z_data, p0=initial_params, bounds=bounds)
                    sigma_y = params[4] * self.pixelsize  # Scale sigma_y with pixel size
                    marker_sigma_y_values.append(sigma_y)
                except:
                    marker_sigma_y_values.append(np.nan)

            sigma_y_values.append(marker_sigma_y_values)
        return sigma_y_values
    '''

    def match_markers(self, ref, tar):
        # match keypoints of target plane to reference plane
        #matches = feature.match_descriptors(ref, tar, max_distance=20, cross_check=True) 
        matches = skim.feature.match_descriptors(ref, tar, cross_check=True) 
        ref_match = ref[matches[:,0]]
        tar_match = tar[matches[:,1]]
        return ref_match, tar_match


    def load_data(self):
        return skim.io.imread(os.path.join(self.filepath, self.filename))

    def get_filename(self):
        
        if not os.path.isdir(self.root_path):
            file_path = filedialog.askopenfile(title='Select registered bead stack', filetypes =[('Registered bead stack', '*.tif')])
            fname = file_path.name 
            fname = os.path.basename(file_path.name)
            fpath = os.path.dirname(file_path.name)
        else:
            calpath = os.path.join(self.root_path, 'cal_data')
            filename = [f for f in os.listdir(calpath) if f.endswith(self.data_ext)]
            
            if len(filename)==1:
                fname = filename[0]
                fpath = calpath
            else:
                file_path = filedialog.askopenfile(title='Select registered bead stack', filetypes =[('Registered bead stack', '*.tif')])
                fname = file_path.name 
                fname = os.path.basename(file_path.name)
                fpath = os.path.dirname(file_path.name)
               

        return fpath, fname
    

    def locs_from_2d(self, mip):
        Nx, Ny = mip.shape
        sigma = self.gauss_sigma
        mip_filt = skim.filters.difference_of_gaussians(mip, low_sigma=sigma)
        # find local peaks, use situational threshold
        # clean up locs from the edges 
        th = np.std(mip_filt)*2 # minval local max
        
        locs = skim.feature.peak_local_max(mip_filt, min_distance=15, threshold_abs = th)
    
        # consolidate by removing locs from the borders
        markForDeletion = []
        for i in range(locs.shape[0]-1):
            if (locs[i][1] <= (self.rr+1)) \
            or (locs[i][1] >= Nx-(self.rr+1)) \
            or (locs[i][0] <= (self.rr+1)) \
            or (locs[i][0] >= Ny-(self.rr+1)):
                markForDeletion = np.append(markForDeletion,i)

        #And delete these indeces from the array
        markForDeletion = np.int_(markForDeletion)
        locs = np.delete(locs,markForDeletion,axis=0)
        # append z position in the stack to the loc
        locs = np.append(locs, np.zeros((locs.shape[0],1), dtype=np.uint16), axis=1)
        return locs
    



def find_focal_plane(stack):
    """
    Finds the focal plane in a 3D stack (z, x, y) based on the second spatial derivative.

    Parameters:
    - stack (numpy.ndarray): The 3D image stack with shape (z, x, y).

    Returns:
    - int: The index of the focal plane in the z-axis.
    """
    # Ensure the stack is a numpy array
    stack = np.asarray(stack)
    
    # Compute the Laplacian for each z-plane
    #laplacian_sum = np.array([np.sum(np.abs(scp.ndimage.laplace(plane))) for plane in stack])
    #laplacian_sum = np.array([np.sum(scp.ndimage.laplace(plane)) for plane in stack])
    laplacian_sum = np.array([np.max(plane) for plane in stack])
    
    # Find the z-plane with the maximum sum of Laplacian values
    focal_plane_index = np.argmax(laplacian_sum)
    
    return focal_plane_index

def phasor_localise(roi, rr):
    #Perform 2D Fourier transform over the complete ROI
    roi_f = np.fft.fft2(roi)
    xangle = np.arctan(roi_f[0,1].imag/roi_f[0,1].real) - np.pi
    #Correct in case it's positive
    if xangle > 0:
        xangle -= 2*np.pi
    #Calculate position based on the ROI radius
    xpos = abs(xangle)/(2*np.pi/(rr*2+1))+0.5

    #Do the same for the Y angle and position
    yangle = np.arctan(roi_f[1,0].imag/roi_f[1,0].real) - np.pi
    if yangle > 0:
        yangle -= 2*np.pi
    ypos = abs(yangle)/(2*np.pi/(rr*2+1))+0.5

    return (xpos,ypos)



###################################
###################################
###################################
###################################
###################################
###################################
###################################

import numpy as np
from scipy.optimize import curve_fit

def gaussian_2D(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """2D Gaussian function."""
    x, y = xy
    exp_term = np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))
    return amplitude * exp_term + offset

def fit_negative_1D_gaussian(x, y):
    """Fit a negative 1D Gaussian to a trace."""
    def negative_gaussian(x, amplitude, mean, sigma, offset):
        return offset - amplitude * np.exp(-((x - mean)**2) / (2 * sigma**2))
    
    initial_params = [np.ptp(y), np.mean(x), np.std(x), np.min(y)]
    bounds = ([0, x.min(), 0, 0], [np.inf, x.max(), np.inf, np.max(y)])
    
    try:
        params, _ = curve_fit(negative_gaussian, x, y, p0=initial_params, bounds=bounds)
        #mean_sigma = params[2]  # sigma of the Gaussian
    except:
        #mean_sigma = np.nan
        params = [np.nan, np.nan, np.nan, np.nan]

    return params
    #return mean_sigma



def calculate_slope(sigma_y_values_stack1, sigma_y_values_stack2, dz):
    """Calculate the slope from the difference in sigma_y values."""
    diff_val = np.array(sigma_y_values_stack1) - np.array(sigma_y_values_stack2)
    z_diff = np.arange(diff_val.shape[1]) * dz
    slopes = []

    for diff in diff_val:
        valid_idx = ~np.isnan(diff)  # Ignore NaN values
        if valid_idx.sum() > 1:
            slope, _ = np.polyfit(z_diff[valid_idx], diff[valid_idx], 1)
            slopes.append(slope)
        else:
            slopes.append(np.nan)

    return np.median(slopes)


###########################



def slope_from_biplanePSF(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    calfile = path
    
    # Load data
    f = scp.io.loadmat(calfile)
    psf1 = f['SXY_g']['PSF'][0][0][0]
    psf2 = f['SXY_g']['PSF'][0][0][1]
    dz = f['SXY'][0][0]['cspline'][0]['dz'][0][0]

    # Setup parameters
    p = {
        'zrange': 600,
        'z0': np.mean([f['SXY'][0][0]['cspline'][0]['z0'][0][0],
                       f['SXY'][0][1]['cspline'][0]['z0'][0][0]]),
        'pixelsizey': 108
    }

    zcal = fit2Dgaussian(psf1, p)
    zcal['cal']['slope'] = slope_biplane_cal(psf1, psf2, p)

    # Save results
    filepath, _ = os.path.split(calfile)
    scp.io.savemat(os.path.join(filepath, "zcal.mat"), {'zcal': zcal})

    return zcal

'''
def gaussian_2D(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    exp_term = np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))
    return amplitude * exp_term + offset
'''

def fit2Dgaussian(stack, p):
    height, width, num_slices = stack.shape
    dz = p['dz']

    # Find z0 and set range
    slice_intensities = np.sum(stack, axis=(0, 1))
    z0 = p.get('z0', np.argmax(slice_intensities))
    slice_multiple = round(p['zrange'] / dz)
    slice_start = max(0, z0 - slice_multiple)
    slice_end = min(num_slices, z0 + slice_multiple)

    sigma_y_values = []

    for i in range(slice_start, slice_end):
        slice_data = stack[:, :, i]
        X, Y = np.meshgrid(range(width), range(height))
        xy_data = np.vstack((X.ravel(), Y.ravel()))
        z_data = slice_data.ravel()

        # Initial parameters
        initial_params = [np.max(slice_data), width / 2, height / 2, 5, 5, np.min(slice_data)]
        bounds = (0, [np.inf, width, height, np.inf, np.inf, np.max(slice_data)])
        try:
            params, _ = curve_fit(gaussian_2D, xy_data, z_data, p0=initial_params, bounds=bounds)
            sigma_y_values.append(params[4] * p['pixelsizey'])
        except RuntimeError:
            sigma_y_values.append(np.nan)

    z_positions = np.linspace(-round(num_slices / 2) * dz, round(num_slices / 2) * dz, len(sigma_y_values))
    return {'z': z_positions, 'PSFynm': sigma_y_values}


def slope_biplane_cal(stack1, stack2, p):
    height, width, num_slices = stack1.shape
    dz = p['dz']

    slice_intensities = np.sum(stack1, axis=(0, 1))
    z0 = p.get('z0', np.argmax(slice_intensities))
    slice_multiple = round(p['zrange'] / dz)
    slice_start = max(0, z0 - slice_multiple)
    slice_end = min(num_slices, z0 + slice_multiple)

    sigma_y_values_stack1 = []
    sigma_y_values_stack2 = []

    for i in range(slice_start, slice_end):
        for stack, sigma_values in zip([stack1, stack2], [sigma_y_values_stack1, sigma_y_values_stack2]):
            slice_data = stack[:, :, i]
            X, Y = np.meshgrid(range(width), range(height))
            xy_data = np.vstack((X.ravel(), Y.ravel()))
            z_data = slice_data.ravel()

            # Initial parameters
            initial_params = [np.max(slice_data), width / 2, height / 2, 5, 5, np.min(slice_data)]
            bounds = (0, [np.inf, width, height, np.inf, np.inf, np.max(slice_data)])
            try:
                params, _ = curve_fit(gaussian_2D, xy_data, z_data, p0=initial_params, bounds=bounds)
                sigma_values.append(params[4] * p['pixelsizey'])
            except RuntimeError:
                sigma_values.append(np.nan)

    diff_val = np.array(sigma_y_values_stack1) - np.array(sigma_y_values_stack2)
    z_diff = np.arange(len(diff_val)) * dz
    slope, _ = np.polyfit(z_diff, diff_val, 1)

    plt.figure()
    plt.plot(z_diff, diff_val, 'bo', label='Data')
    plt.plot(z_diff, np.polyval([slope, 0], z_diff), 'r-', label='Linear Fit')
    plt.xlabel('z (nm)')
    plt.ylabel('Diff(Sigma_y^2)')
    plt.title('Linear Fit of Diff(Sigma_y^2)')
    plt.legend()
    plt.grid()
    plt.show()

    return slope



def visualize_focal_markers(stack1, stack2, markers1, markers2, FP):
    """
    Visualize the markers on the focal slice of each stack to provide visual feedback on data quality.
    
    Parameters:
        stack1 (numpy.ndarray): First stack of images (z, y, x).
        stack2 (numpy.ndarray): Second stack of images (z, y, x).
        markers1 (numpy.ndarray): Markers for stack1, shape (n_markers, 2), columns [x, y].
        markers2 (numpy.ndarray): Markers for stack2, shape (n_markers, 2), columns [x, y].
        FP (tuple): Indices of the focal slices for stack1 and stack2, e.g., (z1, z2).
    """
    z1, z2 = FP  # Focal slice indices

    # Create a figure with two subplots: one for each stack
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the focal slice of stack1 with markers
    axes[0].imshow(stack1[z1], cmap='gray')
    axes[0].set_title(f"Stack1 - Focal Slice {z1}")
    axes[0].axis("off")
    for marker in markers1:
        axes[0].plot(marker[1], marker[0], 'ro', markersize=6, label="Marker1")
    #axes[0].legend(loc="upper left")

    # Plot the focal slice of stack2 with markers
    axes[1].imshow(stack2[z2], cmap='gray')
    axes[1].set_title(f"Stack2 - Focal Slice {z2}")
    axes[1].axis("off")
    for marker in markers2:
        axes[1].plot(marker[1], marker[0], 'go', markersize=6, label="Marker2")
    #axes[1].legend(loc="upper left")

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
