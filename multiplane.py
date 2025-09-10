import skimage as skim 
import scipy as scp
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile
import yaml 
from tkinter import filedialog
from tkinter import *
import json
from glob import glob
import cv2
from natsort import natsorted
import h5py

from utils.qpmain import qpmain
from utils.phase_structure import phase_structure
import utils.metadata as meta 



def get_fileID(filename):
    img_id = os.path.splitext(filename)[0]
    img_id = img_id.replace('_metadata', '')
    img_id = img_id.replace('.ome', '')
    return img_id 

def makeFolder(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print("created folder: ", newpath)


class MultiplaneProcess:

    P = {}
    P['NA_ill']=0.27
    P['dz']=620 #nm
    P['dz_stage']=  100 #nm
    P['dt']=30 #ms, default
    P['pxlsize']= 108 #nm
    P['dF_batch']=2000 #frames, framebatch_size default
    P['do_phase']=False #bool whether to calculate phase from brightfield
    P['do_preproc']=True #bool whether to do preprocessing (FOV detection, cal estimation etc)
    P['ncams']=2 #how many detectors used 
    P['nplanes']= 8 # how many planes across all cameras
    P['dpixel']=7 # remove pixels from frame to remove registration artifacts
    P['order_default']= [2,3,0,1] # default order of planes after cropping
    P['flip_cam'] = [False, True] # bool, whether to flip the camera data (assuming there are 2 cameras)
    P['flip_axis'] = 2 # axis along which planes are mirrored
    P['padding'] = -40 # pixels for padding of found FOV
    P['use_projection'] = 'median' # projection type to use for registration, (median, max, min
    P['ref_plane'] = 2 # reference plane to which  affine transform is determined 
    P['apply_transform'] = True # apply the affine transform before saving data
    P['pretranslate'] = False # determine and apply shift based on all loc centroid before determining affine transform  
    P['zrange_psf'] = 1200 # nm +- around fp to save for psf calibration

    file_extensions = [".tif", ".tiff"]
    log = False


    def __init__(self):

        self.filenames = []
        #self.metadata_files = {}
        self.output_path = None
        self.cal_path = None
        self.path = None
        self.meta = {}
        self.cal = {}
        self.is_bead = False
        self.i_corr = True
        self.save_individual = False
        self.deskew_cam = True
        self.mcal = None # multiplane calibration instance
        self.smlcal = None
        self.markers = {}
        self.th_weight = 0.2 # weights for background and otsu thresholding in adaptive thresholding
        self.save_in_subfolders = False # save each plane in a separate subfolder

        #self.path = self.select_data_directory()




    def select_data_directory(self, path = None):
        if path is not None: 
            self.path = path   

        if self.path is None or not os.path.exists(self.path):
            root = Tk()
            root.withdraw()
            self.path = filedialog.askdirectory(title='select data directory')
        #else: 
        #    self.path = path       

        return self.path
    

    def set_logging(self, log):
        self.log = log
    

    def load_calibration(self):
        if not os.path.exists(os.path.join(self.path, 'cal.json')):
            # ask for user input for calibraiton
            root = Tk()
            root.withdraw()
            filepath = filedialog.askopenfile(title='Select calibration data file', filetypes =[('Calibration file', '*.json')])
            fopen = filepath.name
        else:
            fopen = os.path.join(self.path, 'cal.json')

        f = open(fopen)
        self.cal = json.load(f, object_hook=jsonKeys2int) 
        self.check_calibration() 

        return self.cal
        

    def check_calibration(self):
        # check if calibration file contains all relevant information 
        if type(self.cal) is not dict: 
            print(f"Calibration not a dictionary but {type(self.cal)} check inputs")
            return 
        
        if not bool(self.cal): 
            print(f"Calibration file is empty")
        
        validate_keys = ['fovs', 'brightness', 'dz', 'transform', 'order', 'deg']
        missing_keys = []        
        for k in validate_keys: 
            if k not in self.cal.keys():
                missing_keys.append(k)

        if not missing_keys: 
            print(f"No fields missing in calibration, proceeding")
        else:  
            print('Keys missing: ')
            print(', '.join(map(str, missing_keys))) 
            print("Initiating calibration: ")
            self.calibrate()


    def load_data(self):
        # load image data with standard tiffile reader
        return tifffile.imread(os.path.join(self.path, self.filenames[0]), is_ome=False, is_mmstack=False, is_imagej=False)
    

    def create_cal_path(self):
        self.cal_path = os.path.join(self.path, 'cal_data')
        makeFolder(self.cal_path)

    def create_out_path(self):
        self.output_path = os.path.join(self.path, 'reg')
        makeFolder(self.output_path)
             

    def calibrate(self, is_bead = False, i_corr = True): 
        # do calibration on path data (if no bead data) 
        # if not is_bead: is only approximation but better than nothing
        self.is_bead = is_bead
        self.i_corr = i_corr
        # load first dataset (4gb chunk)
        
        # check whether filelist has been filled 
        if not self.filenames:
            self.get_files_with_metadata()
        try:
            image = tifffile.imread(os.path.join(self.path, self.filenames[0]), is_ome=False, is_mmstack=False, is_imagej=False)
        except:
            image = tifffile.imread(os.path.join(self.path, self.filenames[0]), is_ome=False, is_imagej=False)
        print(f"Read image {self.filenames[0]}; size {image.shape}; type {image.dtype}")
        N_img = image.shape   

        if len(N_img) == 3:
            splits = []
            # Split indices dynamically and append to the list
            for i in range(self.P['ncams']):
                splits.append(image[i::self.P['ncams']])

            # reduce arrazs to same length
            min_len = np.min([l.shape[0] for l in splits])
            splits = [l[:min_len,...] for l in splits]

            image = np.array(np.stack(splits, axis=1))
        # write stack properties
        self.cal['steps'] = image.shape[0]
        self.cal['dz_stage'] = self.P['dz_stage']
        self.cal['pxlsize'] = self.P['pxlsize']
        self.cal['ncams'] = self.P['ncams']
        self.cal['fname'] = self.filenames[0]

        # get global ROI from metadata or insert dummy ROI
        self.cal[f'global_roi']={}
        for i in range(int(self.P['ncams'])):
            if 'global_roi' not in self.P.keys():
                self.cal['global_roi'][i] = [0,0,100,100]
                print(f"Inserting dummy global ROI {self.cal['global_roi'][i]}, please set parameters in final file")
            else:    
                self.cal['global_roi'][i] = self.P['global_roi'] ## currently only one ROI available, still to be fixed from camera metadata maybe? 
                print(f"Using global ROI {self.cal['global_roi'][i]} from parameters, might be erroneous due to missing metadata")

        # find bbox and skew angle
        fovs, self.cal['fovs'], self.cal['deg'] = self.adaptiveThreshold(image, n_planes=self.P['nplanes']) 

        #file_convert = fovs.astype(np.uint16)

        if is_bead: 
            # figure out plane order otherwise take default order
            self.update_metadata(get_fileID(self.filenames[0]))
            self.cal['dz'], self.cal['order'], self.mcal = self.estimate_interplane_distance(fovs)
            self.cal['labels'] = self.mcal.dz['labels'] # plane labels
            self.cal['fp'] = self.mcal.dz['fp'] # focal plane
        else: 
            self.cal['order'] = self.P['order_default'] 
            self.cal['dz'] = self.P['dz']
        
        print(f"Using order {self.cal['order']}")
        fps = np.ones(N_img[0])*(N_img[1]/2)
        fps = fps.astype(np.uint16)

        if 'brightness' not in self.cal.keys():
            self.cal['brightness'] = self.estimate_brightess_from_stack(fovs[self.cal['order'],:,:,:]) 

        if self.i_corr:
            fovs[self.cal['order'],:,:,:] = self.apply_brightness_correction(fovs[self.cal['order'],:,:,:]) 

        if 'transform' not in self.cal.keys():
            if is_bead:
                self.mcal.pretranslate = self.P['pretranslate']
                self.cal['transform'], self.cal['transform_error_rmse'], self.markers, self.cal['scaling_factor'] = self.mcal.get_affine_transform_via_quads(fovs[self.cal['order'],:,:,:], self.P['ref_plane'])
                #self.cal['transform'], self.cal['transform_error_rmse'], self.markers, self.cal['scaling_factor'] = self.mcal.get_micrometry_transformation(fovs[self.cal['order'],:,:,:], self.P['ref_plane'])
                #self.mcal.display_transformations()
                # scale RMSE to pixel size (needs to be per plane// dict format)
                #self.cal['transform_error_rmse'] = self.cal['transform_error_rmse']*self.P['pxlsize'] 
            else:
                self.cal['transform'] = self.get_affine_transform(fovs[self.cal['order'],:,:,:]) 
            # restructure transform 
            self.cal['transform'] = self.restructure_transform(self.cal['transform'])

        if self.P['apply_transform']: 
            print("Registration of data...")
            #registered_subimages = self.register_image_stack(fovs[self.cal['order'],:,:,:], self.cal['transform'])
            registered_subimages = self.mcal.apply_transformation(fovs[self.cal['order'],:,:,:], self.cal['transform']) 
        else: 
            registered_subimages = fovs[self.cal['order'],:,:,:]
            print("Registration of data...")
        

        registered_subimages = np.clip(registered_subimages, 0, 2**16-1).astype(np.uint16)
        if len(registered_subimages.shape) == 4:
            axes = 'ZTYX'
        else:
            # axes = 'ZCTYX'
            axes = 'CTZYX'

        ##### OUT PUT FILE WRITING
        self.create_cal_path()
        
        # first write individual planes to disk
        self.cal['zrange_psf'] = self.P['zrange_psf'] 
        num_slices = np.abs(int(self.P['zrange_psf']/self.cal['dz_stage']))
        self.cal['psf_slices'] = 2*num_slices
        
        for i in range(registered_subimages.shape[0]):
            tifffile.imwrite(os.path.join(self.cal_path, f'beads_zcal_ch{i}.tif'), registered_subimages[i,...], 
                    metadata={
                        'axes': axes,
                        'TimeIncrement': self.P['dt']
                    }
                ) 
            
        # then write whole stack to disk

        tifffile.imwrite(os.path.join(self.cal_path, self.filenames[0]), registered_subimages, 
                    metadata={
                        'axes': axes,
                        'TimeIncrement': self.P['dt'],
                        'ZSpacing': self.P['dz']
                    }
                ) 

        # add processing parameters to file
        for k in self.P.keys():
            if k not in self.cal.keys():
                self.cal[k] = self.P[k]


        self.write_calibration()
        self.write_processing()
        self.write_marker_planes(registered_subimages)

        return self.cal 



    def write_processing(self):
        # write processing parameters to file
        with open(os.path.join(self.path, 'processing.yaml'), 'w') as yaml_file:
            yaml.dump(self.P, yaml_file, default_flow_style=False)
        print("Processing parameters written to file")
        

    def restructure_transform(self, transforms):
        """
        For each transform (dict of 4x3 data), move the first element of each 3-element array to the end.

        Args:
            transforms (dict): A dictionary where each value is a list of 4 numpy arrays, 
                            each of shape (3,), representing rows of a transform.

        Returns:
            dict: A new dictionary with the restructured transforms.
        """
        restructured = {}
        for key, rows in transforms.items():
            if len(rows) != 4 or not all(isinstance(row, np.ndarray) and row.shape == (3,) for row in rows):
                #raise ValueError(f"Invalid format for key {key}. Expected 4 arrays of shape (3,).")
                restructured[key] = rows
                print(f"Nothing to restructure in transform {key}, skipping...")
            else:
                new_rows = [np.array([row[1], row[2], row[0]]) for row in rows]
                restructured[key] = new_rows
        return restructured

    #get_psf_slices(registered_subimages.shape, self.cal['fp'][i], num_slices)
    def get_psf_slices(self, stack_range, fp, num_slices):
        slice_start = np.max([0, int(fp - num_slices)])
        slice_end = np.min([stack_range-1, int(fp + num_slices)]) 
        
        d = 2*num_slices - (slice_end-slice_start)

        if d > 0: 
            if slice_end == stack_range-1:
                slice_start -= d
            elif slice_start == 0:
                slice_end += d
            else:
                print("Cant find appropriate size for psf z range")

        return int(slice_start), int(slice_end)

#     def get_plane_order(self, stack):
#        return stack
#    
    def write_calibration(self):        
        #makeFolder(path)
        with open(os.path.join(self.path,'cal.json'), 'w') as yaml_file:
            #yaml.dump(self.cal, yaml_file, default_flow_style=False)
            json.dump(self.cal, yaml_file, cls=NumpyEncoder)

    # write the focal planes in which the markers are identified to a tiff file
    def write_marker_planes(self, stack):        
       for i in range(stack.shape[0]):
       # Write data to HDF5
           #with h5py.File(os.path.join(self.path,f'locs_{i}.hd5f'), "w") as data_file:
           #    data_file.create_dataset(f'locs_{i}.hd5f', data=self.markers[i])
           fp = self.cal['fp'][i] 
           tifffile.imwrite(os.path.join(self.cal_path, f'fp_{i}.tiff'), stack[i,fp,...])     
           print(f"Finished writing marker plane {i}")


    def get_files_with_metadata(self):
        for file in os.listdir(self.path):
            # check only text files
            for ext in self.file_extensions:
                if file.endswith(ext):
                    self.filenames.append(file)
            if file.endswith('_metadata.txt'): 
                #check if metadata file is present and if so if it has an associated image file
                imagefile_specifier = get_fileID(file)
                if f'{imagefile_specifier}.ome.tif' in os.listdir(self.path):
                    self.meta[f'{imagefile_specifier}'] = {'file': file}

    def parse_metafile(self, filename):
        info = meta.openMetadata(os.path.join(self.path,self.meta[filename]['file']))
        header = meta.getHeader(info)
        frame_info = meta.getFrames(info)
        return header, frame_info
    


    def get_metadata(self):
        for k in self.meta.keys():
            self.meta[k], self.meta[k]['frame_info'] = self.parse_metafile(k)
        return self.meta
    
    def update_metadata(self, filename=None):
        if filename is None: 
            filename = list(self.meta.keys())[0]
        self.P['dz_stage']=  np.round(float(self.meta[filename]['z-step_um']), 4)*1000 #nm
        print(f"Updated zstage displacement per frame to: {self.P['dz_stage']:.1f} nm")
        
        # get first frame and read out metadata info 
        frame_keys = list(self.meta[filename]['frame_info'].keys())
        frame_info = self.meta[filename]['frame_info'][frame_keys[0]]['info']
        if 'ROI' in frame_info.keys():
            # if ROI is defined, update global ROI
            roi = frame_info['ROI']
        else:
            print("No ROI defined in metadata, using default global ROI")
            roi = '0-0-100-100' # default ROI, to be adjusted later
        self.P['global_roi'] = [int(i) for i in roi.split('-')]
        print("Updated global ROI to: ", self.P['global_roi'])

        if 'PixelSizeUm' in frame_info.keys():
            self.P['pxlsize'] = float(frame_info['PixelSizeUm'])*1000
            print("Updated pixelsize to: ", self.P['pxlsize'])
    
    
    def estimate_brightess_from_stack(self, stack):
    # stack: z, t, y, x
        z, _,_,_ = stack.shape
        average_brightness = np.empty(shape=(z))
        for i in range(z):
            average_brightness[i] = np.mean(stack[i,::2,:,:].squeeze())
        brightness_factors = [b/np.max(average_brightness) for b in average_brightness]
        b = {i: k for i,k in enumerate(brightness_factors)}
        return b 
    
    def upright_images(self, stack, P=None, log=False):
        datatype = stack.dtype
        angles = np.linspace(-3, 3, 31)
        max_upright_pixels = 0
        best_angle = 0
        minp = np.min(stack, axis=0)
        min_dim = np.argmin(minp.shape)
        if P is None:
            print("Determine skew angle...")
            for angle in tqdm(angles):
                # Rotate the image
                rotated_img = skim.transform.rotate(minp, angle, resize=False, mode="wrap", preserve_range=True)
                # Gaussian smoothing
                smoothed_img = skim.filters.gaussian(rotated_img, sigma=10, preserve_range=True)
                # Canny edge detection
                edges = skim.feature.canny(smoothed_img, sigma=1)

                lines = np.ones_like(smoothed_img) * edges
                img_projection = np.sum(lines, axis=min_dim)
                img_projection = img_projection[2: -3]

                upright_pixels = np.max(img_projection)
                if upright_pixels > max_upright_pixels:
                    max_upright_pixels = upright_pixels
                    best_angle = angle
            if self.log: 
                print(f"Best Angle: {best_angle} degrees")
                print(f"Max Line Count: {max_upright_pixels}")

                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1), plt.imshow(minp, cmap="gray"), plt.title("Original Image")
                plt.subplot(1, 2, 1), plt.imshow(minp, cmap="gray"), plt.title("Original Image")
                plt.subplot(1, 2, 2), plt.imshow(skim.transform.rotate(minp, best_angle, resize=True, mode="wrap"), cmap="gray"), plt.title("Best Detected Angle")
                plt.show()
        else:
            best_angle = P

        print("Rotating by skew angle...")
        limit = [-0.1, 0.1]
        if limit[0] < best_angle < limit[1]:
            rotated_stack = stack
        else:
            rotated_stack = self.rotate_stack(stack, best_angle, datatype)

        return rotated_stack, best_angle

    def rotate_stack(self, stack, best_angle, datatype):
        return scp.ndimage.rotate(stack, best_angle, axes=(1, 2), reshape=False, output=datatype, mode="wrap")

    def adaptiveThreshold(self, stack, n_planes=4, z_axis=0, camera_axis=1, size_estimate=None):
        flip = self.P['flip_cam']
        if 'padding' not in self.P.keys():
            self.P['padding'] = 20

        dim = stack.shape[z_axis]
        remaining_axis = np.linspace(0, len(stack.shape)-1, len(stack.shape)).astype(np.int32)
        remaining_axis = np.delete(remaining_axis, [z_axis, camera_axis])
        Nx, Ny = remaining_axis[0], remaining_axis[1]
        planes_per_cam = int(n_planes/stack.shape[camera_axis])
        if size_estimate is None: 
            size_estimate = stack.shape[Nx]*stack.shape[Ny]/(planes_per_cam)*0.8 ## adjusted from factor 1.2 (estimate should be smaller than fovsize/planes)

        mip = np.median(stack, axis=z_axis)
        fov_props = {}
        angle_props = {}
        max_height, max_width = 0, 0
        for cam in range(stack.shape[camera_axis]):
            fov_props[cam] = {}
            if self.deskew_cam:
                stack[:,cam,:,:], deskew_angle = self.upright_images(stack[:,cam,:,:])
            else:
                deskew_angle = 0
            angle_props[cam] = deskew_angle
            if self.P['use_projection'] == 'median':
                mip = np.median(stack[:,cam,:,:], axis=z_axis) 
            if self.P['use_projection'] == 'min':
                mip = np.min(stack[:,cam,:,:], axis=z_axis) 
            else:
                mip = np.max(stack[:,cam,:,:], axis=z_axis) 
            #mip = skim.filters.gaussian(mip, sigma=5, preserve_range=True)
            th = skim.filters.threshold_otsu(mip.ravel())#np.quantile(mip.ravel(), 0.3)
            max_val= np.max(mip.ravel())

            if self.log: 
                plt.ion()

            print(f"Adaptive thresholding cam {cam}..")


            # try it with continous erosion and a background estimate as threshold
            th = skim.filters.threshold_otsu(mip.ravel())#np.quantile(mip.ravel(), 0.3)
            bkg = np.mean([np.median([mip[:,0].ravel(), mip[:,-1].ravel()]),np.median([mip[0,:].ravel(), mip[-1,:].ravel()])])
            
            #bkg= (bkg*w[0]+th*w[1])/np.sum(w)
            th_combined = bkg + self.th_weight * (th-bkg)
            props, mask = self.erode_image(mip, size_estimate, th_combined, planes_per_cam)

            if self.log:
                plt.imshow(mask)
                plt.show()

            for idx, p in enumerate(props):
                fov_props[cam][idx] = list(p.bbox)
                if p.bbox[2]-p.bbox[0] > max_width:
                    max_width = p.bbox[2]-p.bbox[0]
                if p.bbox[3]-p.bbox[1] > max_height:
                    max_height = p.bbox[3]-p.bbox[1]

            # apply some padding if erosion removed part of the FOV
            max_width= np.min([max_width+self.P['padding'], mip.shape[0]]).astype(int) 
            max_height= np.min([max_height+self.P['padding'], mip.shape[1]]).astype(int)

        # consolidate bbox size
        image_crops = np.empty(shape=(n_planes, dim, max_width, max_height))
        for cam_idx, cam_props in tqdm(fov_props.items(), "FOV size consolidation"):
            for planes_idx, planes_bbox in cam_props.items():
                planes_bbox = self.adjust_bbox(mip.shape, planes_bbox, (max_width, max_height))
                fov_idx = int(cam_idx*planes_per_cam+planes_idx)
                image_crops[fov_idx,:,:,:] = np.expand_dims(self.crop_bbox(stack[:,cam_idx,:,:].squeeze(), planes_bbox), axis=0)

                if flip[cam_idx]:
                    # causes memory issues due to float conversion, do the flipping in place iteratively? 
                    for t in range(image_crops.shape[1]):
                        image_crops[fov_idx,t,:,:] = np.flip(np.squeeze(image_crops[fov_idx,t,:,:]), axis=1) # axis change?

        if self.log: 
            fig, ax = plt.subplots(1, n_planes, figsize=(n_planes*3, 3))
            for t in range(n_planes):
                ax[t].imshow(np.median(image_crops[t,:,:,:], axis=0))
                ax[t].set_title(f'FOV_{t}')
                ax[t].axis("off")
            fig.set_tight_layout(True) 
            plt.show()

        return image_crops.astype(np.uint16), fov_props, angle_props




    def erode_image(self, mip, size_estimate, th, n_planes):
        # Step 1: Threshold the image to create a binary mask
        binary_mask = mip > th
        binary_mask = np.logical_and(np.ones(mip.shape), binary_mask > 0)
        binary_mask = binary_mask.astype(int)
        # Factor for size tolerance
        size_min = size_estimate * 0.7
        size_max = size_estimate * 1.3
        
        # Step 2: Iteratively apply erosion until we get the desired number of targets with the desired size
        iteration = 0
        fail = False
        while True:
            # Erode the mask
            eroded_mask = skim.morphology.binary_erosion(binary_mask, skim.morphology.square(3))
            if self.log:
                plt.imshow(eroded_mask)
                plt.show()
            # Label connected components
            labeled_mask = skim.measure.label(eroded_mask)
            
            # Measure the properties of the labeled regions
            regions = skim.measure.regionprops(labeled_mask)
            
            # Filter regions by size
            if iteration < 10:
                valid_regions = [r for r in regions if size_min <= r.area_bbox <= size_max]
            else:
                valid_regions = [r for r in regions]
            
            # Check if the number of valid regions matches n_planes
            if len(valid_regions) == n_planes:
                break
            
            # If eroded_mask becomes empty (no targets left), break the loop
            if not eroded_mask.any():

                if fail:
                    print(f"Failed to find {n_planes} targets with the desired size after {iteration} iterations. Consider adjusting parameters.")
                    break
                else: 
                    th = th/2
                    binary_mask = mip > th
                    binary_mask = np.logical_and(np.ones(mip.shape), binary_mask > 0)
                    binary_mask = binary_mask.astype(int)
                    fail = True
                    continue
            
            # Update the mask for the next iteration
            binary_mask = eroded_mask
            
            iteration += 1
        
        # Step 3: Return the final mask and number of iterations taken
        return valid_regions, eroded_mask



    def filter_fov_size(self, fovs, s):
        # fovs: list of potential fovs
        # s: size estimate (lower bound, upper bound)
        out = []
        for f in fovs: 
            if s[0] < f.area_bbox < s[1]:
                out.append(f)
        return out


    def crop_with_parameters(self, stack, P, n_planes=4, z_axis=0, camera_axis=1):
        flip = self.P['flip_cam'] 
        dim = stack.shape[z_axis]
        remaining_axis = np.linspace(0, len(stack.shape)-1, len(stack.shape), dtype=int)
        remaining_axis = np.delete(remaining_axis, [z_axis, camera_axis])
        Nx, Ny = remaining_axis[0], remaining_axis[1]
        planes_per_cam = int(n_planes/stack.shape[camera_axis])

        fov_props = P["fovs"]
        f0 = fov_props[0][0]
        max_width, max_height = f0[2]-f0[0], f0[3]-f0[1]
        if self.deskew_cam:
            for cam in range(stack.shape[camera_axis]):
                stack[:,cam,:,:], _ = self.upright_images(stack[:,cam,:,:], P["deg"][cam])

        image_crops = np.empty(shape=(n_planes, dim, max_width, max_height))

        for cam_idx, cam_props in fov_props.items():
            for planes_idx, planes_bbox in cam_props.items():
                fov_idx = int(cam_idx*planes_per_cam+planes_idx)
                image_crops[fov_idx,:,:,:] = np.expand_dims(self.crop_bbox(stack[:,cam_idx,:,:].squeeze(), planes_bbox), axis=0)

                if flip[cam_idx]:
                    image_crops[fov_idx,:,:,:] = np.flip(np.squeeze(image_crops[fov_idx,:,:,:]), axis=self.P['flip_axis'])

        return image_crops


    def adjust_bbox(self, shape, bbox, bbox_size):
        #assert bbox[2]-bbox[0] <= bbox_size[0] <= shape[0], f"Dimension 0 of bounding box {bbox} out of range for bbox_size {bbox_size} and image shape {shape}"
        #assert bbox[3]-bbox[1] <= bbox_size[1] <= shape[1], f"Dimension 1 of bounding box {bbox} out of range for bbox_size {bbox_size} and image shape {shape}"

        for i in range(len(shape)):

            if bbox_size[i] == shape[i]:
                bbox[i] = 0
                bbox[i+2] = shape[i]
            else:
                diff = bbox_size[i] - (bbox[i+2]-bbox[i])
                c0, c1 = bbox[i]-diff/2, bbox[i+2]+diff/2
                d = 0 # final difference to check whether bbox fits into image
                # can not both be true due to input check
                if c0 < 0: 
                    d = c0
                elif c1 > shape[i]:
                    d = c1 - shape[i]
                # min max conditions shouldnt be necessarz here, check again
                bbox[i] = int(np.max([np.rint(c0 - d), 0]))
                bbox[i+2] = int(np.min([np.rint(c1 - d), shape[i]]))

                # safety check cause im a fucking idiot and cant get the stupid numpy rounding rules right (even, odd numbers and .5)
                # so,metimes bbox is one off, correct that
                dd = (bbox[i+2]-bbox[i])-bbox_size[i]
                if dd>0:
                    bbox[i+2] -= dd
                elif dd<0:
                    if bbox[i+2] < shape[i]:
                        bbox[i+2] -= dd
                    else:
                        bbox[i] += dd

        return bbox

    def crop_bbox(self, stack, bb):
        return stack[:,bb[0]:bb[2], bb[1]:bb[3]]

#     def save_yaml(self, stack, path):
#        for i in range(stack.shape[1]):
#            params = {}
#            filename = os.path.join(path, f'Plane_{i}')
#            params["Plane"] = i
#            params["directory"] = os.path.join(filename, "")
#            self.metadata_files[i] = filename
#            with open(filename + ".yaml", "w") as file:
#                yaml.dump(params, file)
#
#     def save_stack(self, stack):
#        for i in range(stack.shape[0]):
#            tifffile.imwrite(self.metadata_files[i] + ".tiff", stack[i])
#
#     def get_average_transform_via_xcorr(self, stack, fp):
#        # stack: z, t, y, x 
#        # fp: focal planes (int), shape: (z,1)
#        z, t,_,_ = stack.shape
#        transforms = np.empty(shape=(z-1, 2))
#        eval_points = min(t, 5)
#        eval_points_delta  = np.linspace(max(-int(t/2), -20), min(int(t/2), 20), num=min(eval_points, 7), dtype=int)
#        upsample = 100
#        for p in range(z-1): 
#            #eval_plane = int(np.round(np.mean([fp[p], fp[p+1]])))
#            #iterate points in bead stack
#            eval_points_shift = np.empty(shape=(eval_points, 2))
#            for z_point in range(eval_points):
#            # pixel level precision first
#                shift, _,_ = skim.registration.phase_cross_correlation(stack[0,fp[0]+eval_points_delta[z_point],:,:], stack[p+1,fp[p+1]+eval_points_delta[z_point],:,:], upsample_factor=upsample) # evaluate at focal plane
#                eval_points_shift[z_point] = shift/upsample
#            transforms[p] = np.mean(eval_points_shift, axis=0)
#        return transforms
#
#
#     def get_average_transform_via_SIFT(self, stack):
#        # stack: z, t, y, x 
#        # fp: focal planes (int), shape: (z,1)
#        z, t, y, x = stack.shape
#        transforms = np.empty(shape=(z-1, 256))
#        descriptor_extractor = skim.feature.SIFT(upsampling=2, c_dog=0.01, sigma_in=0.1) #BRIEF() # SIFT()
#
#        mips = np.max(stack, axis=1)  # np.empty(shape=(z,y,x))
#        descriptors = {}
#
#        descriptor_extractor.detect_and_extract(mips[0,...])
#        #descriptor_extractor.extract(mips[0,...])
#        #keypoints = descriptor_extractor.keypoints
#        descriptors[0] = descriptor_extractor.descriptors
#
#        for p in range(1, z): 
#
#            #iterate projections for their descriptors and match to first plane
#            descriptor_extractor.detect_and_extract(mips[p,...])
#            #descriptor_extractor.extract(mips[p,...])
#            #keypoints = descriptor_extractor.keypoints
#            descriptors[p] = descriptor_extractor.descriptors
#            
#            m = skim.feature.match_descriptors(descriptors[0], descriptors[p], max_ratio=0.6, cross_check=True) # matching indices in descriptor sets
#            # pixel level precision first
#            transforms[p] = skim.transform.estimate_transform('affine', descriptors[0][m[:,0]], descriptors[p][m[:,1]])
#            
#        return transforms
#
#
    def transform_stack(self, stack, transform):
        # stack: z, t, y, x 
        # transform:  (z,2) (xy shift vector)
        z, _, _, _ = stack.shape
        outer = tqdm(total=z-1, desc='Image plane', position=1)
        for ip in range(z-1):
            stack[ip+1,:,:,:] = self.shift_via_fft(stack[ip+1,:,:,:].squeeze(), transform[ip])
            outer.update(1)
        return stack

#     def shift_via_fft(self, stack, transform):
#        # stack: t, y, x
#        # transform: (x,y)
#        t, _, _ = stack.shape
#        inner = tqdm(total=t, desc='timepoint', position=0)
#        for img in range(t):
#            transformed_img = scp.ndimage.fourier_shift(input=np.fft.fftn(stack[img,:,:]), shift=transform)
#            stack[img,:,:] = np.fft.ifftn(transformed_img)
#            inner.update(1)
#        return stack
#
    def estimate_interplane_distance(self, stack):
        from multiplane_calibration import MultiplaneCalibration
        cal = MultiplaneCalibration()
        cal = cal.set_zstep(self.P['dz_stage'])
        res = cal.estimate_interplane_distance(stack)
        self.create_cal_path()
        self.write_figure(cal.figs['dz'], self.cal_path, "interplane_distance", '.svg')
        self.write_figure(cal.figs['dz'], self.cal_path, "interplane_distance", '.png')

        return cal.dz['dz'], cal.order, cal


    def write_figure(self, f, outpath, fname, filetype):
        # f: figure handle
        # path: output path
        # fname: filename

        output_name = os.path.join(outpath, fname+filetype)
        makeFolder(outpath)
        f[0].savefig(output_name, dpi = 600, bbox_inches="tight", pad_inches=0.1, transparent=True)    
        print(f"Finished writing {output_name}")
#
#
    def apply_brightness_correction(self, image):
        for p in range(image.shape[0]):
            image[p,...] = np.divide(image[p,...], self.cal['brightness'][p])
        return image

    def execute(self):
        #for self.root, _, self.filenames in os.walk(self.path):
        #    self.filenames = [os.path.join(self.root, file) for file in self.filenames if file.endswith(tuple(self.file_extensions))]
        if not self.filenames:
            self.get_files_with_metadata()
        self.filenames.sort()
        print("Data Directory:", self.path)

        self.check_calibration()

        run=True
        filecounter=0
        idx=0 # batchindex, counter
        f=0 #framecounter
        ncams = self.P['ncams']

        file_specifier = get_fileID(self.filenames[filecounter])
       
        for image in read_tiff_series_batch(self.path, batch_size=self.P['dF_batch'], n_cams=self.P['ncams']):
            ### 
            idx+=1
            N_img = image.shape 
            # apply deg rotation and fov cropping
            fovs = self.crop_with_parameters(image, self.cal, n_planes=self.P['nplanes']) 


            fovs[self.cal['order'],:,:,:] = self.apply_brightness_correction(fovs[self.cal['order'],:,:,:]) 

            if self.P['apply_transform']: 
                print("Registration of data...")
                registered_subimages = self.register_image_stack(fovs[self.cal['order'],:,:,:], self.cal['transform'])
            else: 
                registered_subimages = fovs[self.cal['order'],:,:,:]

            # clean up values outside 16bit tiff range    
            registered_subimages = np.clip(registered_subimages, 0, 2**16-1).astype(np.uint16)
            if len(registered_subimages.shape) == 4:
                if self.save_individual:
                    axes = 'TYX'           
                else:
                    axes = 'ZTYX'
            else:
                # axes = 'ZCTYX
                if self.save_individual:
                    axes = 'CTYX'       
                else:
                    axes = 'CTZYX'

            # save stack
            self.create_out_path()

            print(f"Writing data to {self.output_path}")

            clean_file_specifier = file_specifier.replace("MMStack", "mm")

            if self.save_individual:
                for plane in tqdm(range(registered_subimages.shape[0]), desc="Plane"): 
                    # do yhou want to save data of each plane in a different folder?
                    if self.save_in_subfolders:
                        plane_path = os.path.join(self.output_path, str(plane))
                        makeFolder(plane_path)
                    else: # save all planes in the same folder
                        plane_path = self.output_path

                    tifffile.imwrite(os.path.join(plane_path, f"{clean_file_specifier}_f{idx}_pl{plane}.tif"), registered_subimages[plane], 
                        metadata={
                            'TimeIncrement': self.P['dt'],
                            'ZSpacing': self.P['dz']
                        }
                    ) 

            else:
                tifffile.imwrite(os.path.join(self.output_path, f"{clean_file_specifier}_f{idx}.ome.tif"), registered_subimages, 
                    metadata={
                        'axes': axes,
                        'TimeIncrement': self.P['dt'],
                        'ZSpacing': self.P['dz']
                    }
                ) 
        print(f"Finished processing {self.path}")

#########################################
# single molecule localisation calibration
#########################################


    def calibrate_sml(self):
        from smlm_calibration import smlm_calibration
        if self.mcal is None:
            markers = None
        else:
            markers = self.mcal.markers
            
        self.smlcal = smlm_calibration(self.path, markers, self.P["ref_plane"], self.P["dz_stage"], self.P["pxlsize"])
        cal = self.smlcal.biplane_calibration()
        cal['fp'] = [0.0] + [np.sum(self.cal['dz'][:p+1]) for p in range(len(self.cal['dz']))]
        cal_outer = {"zcal": cal}
        outname = os.path.join(self.path, "zcal.mat")
        scp.io.savemat(outname, cal_outer)

        return cal_outer


#########################################
#TRANFORMS
#########################################

    def get_affine_transform(self, stack):
        # stack: z, t, y, x 
        # fp: focal planes (int), shape: (z,1)
        z, t, y, x = stack.shape
        transforms = np.empty(shape=(z,2,3))
        for p in range(z): 
            # pixel level precision first
            transforms[p] = self.find_affine_transformation(stack[self.P['ref_plane']], stack[p])
        return transforms

    '''
    def find_affine_transformation(self, image_stack, reference_stack):

        tar = np.max(image_stack, axis=0)
        ref = np.max(reference_stack, axis=0)

        
        # Use ORB to find keypoints and descriptors
        orb = cv2.ORB_create(scaleFactor=1.2)
        #orb = cv2.SURF_create()
        keypoints1, descriptors1 = orb.detectAndCompute(tar.astype(np.uint8), None)
        keypoints2, descriptors2 = orb.detectAndCompute(ref.astype(np.uint8), None)

        # Use BFMatcher to find matches
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract locations of matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find the affine transformation matrix
        #matrix, mask = cv2.estimateAffine2D(src_pts, dst_pts)
        matrix= cv2.getAffineTransform(src_pts, dst_pts)

        return matrix
        '''


    def apply_affine_transformation(self, matrix, img):
        # Apply the affine transformation
        matrix = np.array(matrix, dtype=np.float32)
        #transformed_img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        transformed_img = cv2.warpAffine(img, matrix[:2,:], (img.shape[1], img.shape[0]))
        return transformed_img
    
    def register_image_stack(self, stack, matrix):
        # stack: z, t, y, x 
        # fp: focal planes (int), shape: (z,1)
        z, t, y, x = stack.shape
        # Prepare an array to store transformed images
        #transformed_stack = np.zeros_like(stack)

        for i in tqdm(range(z), desc=" Image plane", position=0):
            # Process each image in the stack
            for j in tqdm(range(t), desc=" Timepoint", position=1, leave=False):
        
            # Apply the affine transformation
                transformed_img = self.apply_affine_transformation(matrix[i], stack[i, j,...])
                # Store the transformed image
                stack[i,j] = transformed_img

        return stack

#######################################################
#END OF CLASS
#######################################################

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, np.generic):
            return obj.tolist()
        return super().default(obj)
    


def jsonKeys2int(x):
    if isinstance(x, dict):
        return {(int(k) if k.isnumeric() else k):v for k,v in x.items()}
    return x


def read_tiff_series_batch(folder_path, batch_size=100, n_cams=2, file_extension='tif'):
    """
    Read an image series batch-wise from multiple TIFF files in a folder and allocate to a 4D array.
    
    :param folder_path: Path to the folder containing TIFF files.
    :param batch_size: Number of frames to read in each batch.
    :param n_cams: Number of channels (e.g., cameras) for each frame.
    :param file_extension: File extension for TIFF files, default is 'tif'.
    :yield: A 4D NumPy array of shape (batch_size, n_cams, height, width).
    """
    # Get all TIFF files in the folder
    #tiff_files = sorted(glob(os.path.join(folder_path, f'*.{file_extension}')))
    tiff_files = natsorted(glob(os.path.join(folder_path, f'*.{file_extension}')))
    
    current_batch = []  # To store images for the current batch
    
    # Iterate through each TIFF file
    for tiff_file in tiff_files:
        with tifffile.TiffFile(tiff_file) as tif:
            total_pages = len(tif.pages)
            
            # Iterate through pages of the current TIFF file
            for i in range(total_pages):
                try:
                    # Read the current page as a NumPy array (skip reading metadata)
                    image = tif.pages[i].asarray()
                    current_batch.append(image)
                except UnicodeDecodeError:
                    print(f"UnicodeDecodeError on page {i} of file {tiff_file}, skipping metadata.")

                # If the batch size is reached, process and yield the batch
                if len(current_batch) == batch_size * n_cams:
                    # Reshape into 4D array: (batch_size, n_cams, height, width)
                    batch_array = np.array(current_batch)
                    batch_array = batch_array.reshape(batch_size, n_cams, *batch_array.shape[1:])
                    yield batch_array
                    current_batch = []  # Reset the batch after yielding
        
    # If there are remaining images after all files are processed, yield them as a batch
    if current_batch:
        remaining_size = len(current_batch) // n_cams

        try: 
            batch_array = np.array(current_batch)
            batch_array = batch_array.reshape(remaining_size, n_cams, *batch_array.shape[1:])
        except ValueError:
            *batch_array, _ = batch_array
            batch_array = np.array(current_batch)
            batch_array = batch_array.reshape(remaining_size, n_cams, *batch_array.shape[1:])
        
        yield batch_array

