import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import skimage.filters as skfilt
from skimage import feature, io
from skimage.transform import AffineTransform, estimate_transform
from scipy.spatial import cKDTree
from scipy import ndimage, spatial
from scipy.optimize import curve_fit
from tqdm import tqdm 
from matplotlib.colors import LinearSegmentedColormap
import warnings
import os
import cv2
import quads
import math
warnings.filterwarnings('ignore')



'''
cdict = {'violet': "#960792ff",
             'orange': "#ff3b3bff", 
             'yellow': "#b28900ff", 
             'blue': "#009bb2ff", 
             'aquamarin':'#529b9c', 
             'brown':'#994133'}
'''
# diverging colortheme, 8 classes
'''
cdict = {0:'#d73027', 
         1: '#f46d43',
         2: '#fdae61',
         3: '#fee090',
         4: '#e0f3f8',
         5: '#abd9e9',
         6: '#74add1',
         7: '#4575b4'}
'''
cdict = {
    0:  '#a50026',
    1:  '#d73027',
    2:  '#f46d43',
    3:  '#fdae61',
    4:  '#fee090',
    5:  '#ffffbf',
    6:  '#e0f3f8',
    7:  '#abd9e9',
    8:  '#74add1',
    9:  '#4575b4',
    10: '#313695',
    11: '#2c7bb6',
    12: '#00a6ca',
    13: '#00ccbc',
    14: '#90eb9d',
    15: '#ffffcc'
}

global clist
clist = list(cdict.values())

class MultiplaneCalibration:
    # calibration on diffraction limited bead data
    # fovs of subimages already selected, transformation not yet calculated
    # input: stack, (plane, z, x, y) of bead data

    def __init__(self):
        self.dz = {}
        self.pos_sr = {}
        self.order = []
        self.beads = {}
        self.log = True
        self.tracks={}
        self.figs = {}
        self.pretranslate = False
        self.check_tilt = False
        self.kd = {}    # distance trees
        self.quad = {}  # quads 
        self.quad_match = {}  # matched quads  
        self.scaling_factor = {} # datastore for estimated magnification factor
        self.check_magnification = True # calculate sparse distance tree and evaluate against others
        self.beadID = {} # bead candidates found in maximum intensity projection
        self.markers = {} # SR loclaised beads in MIP for tranformation finding
        self.transform = {}
        self.transform_error = {}
        #processing parameters
        self.pp = {'gauss_sigma': 1.5, # sigma for DoG gaussian kernel
                   'roi' : 10, # roi radius around peak, to delete locs near edges 
                   'frame_min' : 15, # min amount of consecutive frames to consider it a bead trace
                   'd_max' : 5, # maximum distance of locs in consecutive frames to be considered belonging to the same trace 
                   'zstep': 10, # default stage zstep size for coregistration nd PSF fitting
                   'min_size': 5.0, # min size for quad finding in distance trees 
                   'max_size': 300.0, # max size for quad finding in distance trees
                   'tolerance': 0.01,  # tolerance for quad acceptance
                    'max_neighbours' : 5} #distance tree neighbours, was == 10


#     def setlog(self, log=bool):
#        self.log = log
#        return self
#
#
    def estimate_interplane_distance(self, stack):
        print('Estimating interplane distance..')
        planes = stack.shape[0]
        self.pp['planes'] = planes
        self.pp['stack_height'] = stack.shape[1]
        self.pp['stack_sx'] = stack.shape[-2]
        self.pp['stack_sy'] = stack.shape[-1]
        pos_candidate_all = {}
        outer = tqdm(total=planes, desc='Finding peak candidates', position=0)
        for p in range(planes):
            outer.update(1)
            #pos_candidate_all[p] = self.find_candidate_positions(stack[p,...])
            self.beadID[p] = self.find_candidate_positions_in_projection(stack[p,...])

            pos_candidate_all[p] = self.expand_positions_in_z(self.beadID[p])

        # create a list of potential positions from bead candidates 
        outer = tqdm(total=planes, desc='SR-localising peaks', position=0)
        for p in range(planes):
            self.pos_sr[p]= self.localise_candidates(stack[p,...], pos_candidate_all[p])
            outer.update(1)
        
        # Identifying beads in z
        outer = tqdm(total=planes, desc='Tracking beads in z', position=0)
        for p in range(planes):
            outer.update(1)
            self.beads[p] = self.track_locs_in_z(self.pos_sr[p], self.beadID[p])
            self.beads[p] = self.clean_up_tracks(self.beads[p])

        outer = tqdm(total=planes, desc='Convert datastructure', position=0)
        for p in range(planes):
            outer.update(1)
            self.tracks[p] = self.convert_dict_to_array(self.beads[p])
        
        if self.check_tilt:
            outer = tqdm(total=planes, desc='Estimating tilt', position=0)
            for p in range(planes):
                self.beads_dxy[p] = self.estimate_dxy(self.beads[p])

        print('Determining relative z-distances and order')
        self.dz, self.order = self.get_dz()
        print(self.dz)
        return self.dz, self.order, self


    def set_zstep(self, zstep):
        self.pp['zstep'] = zstep
        return self
        
    def get_figures(self):
        return self.figs


    def convert_dict_to_array(self, dict):
        vals = dict.values()
        return np.array(list(vals))

    def find_candidate_positions_in_projection(self, stack):
        assert len(stack.shape)==3, "stack has wrong dimensions"
        mip = np.max(stack, axis=0)
        locs = self.locs_from_2d(mip)
        return locs
    

    def locs_from_2d(self, mip):
        Nx, Ny = mip.shape
        sigma = self.pp['gauss_sigma']
        mip_filt = skfilt.difference_of_gaussians(mip, low_sigma=sigma)
        # find local peaks, use situational threshold
        # clean up locs from the edges 
        th = np.std(mip_filt)*2 # minval local max
        
        locs = feature.peak_local_max(mip_filt, min_distance=7, threshold_abs = th)
    
        # consolidate by removing locs from the borders
        markForDeletion = []
        for i in range(locs.shape[0]-1):
            if (locs[i][1] <= (self.pp['roi']+1)) \
            or (locs[i][1] >= Nx-(self.pp['roi']+1)) \
            or (locs[i][0] <= (self.pp['roi']+1)) \
            or (locs[i][0] >= Ny-(self.pp['roi']+1)):
                markForDeletion = np.append(markForDeletion,i)

        #And delete these indeces from the array
        markForDeletion = np.int_(markForDeletion)
        locs = np.delete(locs,markForDeletion,axis=0)
        # append z position in the stack to the loc
        locs = np.append(locs, np.zeros((locs.shape[0],1), dtype=np.uint16), axis=1)
        return locs

    def find_candidate_positions(self, stack):
        assert len(stack.shape)==3, "stack has wrong dimnensions"
        z_pos, Nx, Ny = stack.shape
        # stack: z,x,y beadstack in a single subimage
        inner = tqdm(total=z_pos, desc='Filtering', position=0)
        filt = np.empty_like(stack, dtype=np.float32)
        # apply dog filter to amplify spatial derivatives
        sigma = self.pp['gauss_sigma']
        for z in range(z_pos):
            filt[z,:,:] = skfilt.difference_of_gaussians(stack[z,:,:], low_sigma=sigma)
            inner.update(1)
        
        # find local peaks, use situational threshold
        # clean up locs from the edges 
        #loc_peaks = np.empty((1,3)) # container of final locs
        inner = tqdm(total=z_pos, desc='Peak finding', position=0)
        for z in range(z_pos):
            th = np.std(filt[z,:,:])*2 # minval local max
            locs = feature.peak_local_max(filt[z,:,:], threshold_abs = th)

            # consolidate by removing locs  from the borders
            markForDeletion = []
            for i in range(locs.shape[0]-1):
                if (locs[i][0] <= (self.pp['roi']+1)) \
                or (locs[i][0] >= Nx-(self.pp['roi']+1)) \
                or (locs[i][1] <= (self.pp['roi']+1)) \
                or (locs[i][1] >= Ny-(self.pp['roi']+1)):
                    markForDeletion = np.append(markForDeletion,i)

            #And delete these indeces from the array
            markForDeletion = np.int_(markForDeletion)
            locs = np.delete(locs,markForDeletion,axis=0)
            # append z position in the stack to the loc
            locs = np.append(locs, np.ones((locs.shape[0],1), dtype=np.uint16)*z, axis=1)

            inner.update(1)

            if z == 0: 
                loc_peaks = locs
            else:
                loc_peaks = np.append(loc_peaks, locs, axis=0)

        return loc_peaks
         


    # localise position accurately with phasor method
    def localise_candidates(self, stack, pos_candidate):
        assert len(stack.shape)==3, "stack has wrong dimnensions"
        assert pos_candidate.shape[0] > 0, "No beads found"
        rr = self.pp['roi']
        z_pos, Nx, Ny = stack.shape
        pos_sr = pos_candidate.copy().astype(np.float32)
        pos_sr = np.append(pos_sr, np.zeros((pos_sr.shape[0], 1), dtype=np.float32), axis=1)        
        skip_counter = 0    

        for peak in range(pos_candidate.shape[0]):
            l = pos_candidate[peak]
            roi = stack[l[2], int(l[0]-rr):int(l[0]+rr), int(l[1]-rr):int(l[1]+rr)]
            
            if roi.shape[0]!=roi.shape[1] or roi.shape[0]==0:
                skip_counter+=1
                #print(f"Skipping peak {peak}, irregular shape: {roi.shape}")
                continue

            # sr position in roi crop 
            roi_pos = list(self.phasor_localise(roi))

            # update to global coordinates
            sr_pos = [l[0]-rr+roi_pos[0], l[1]-rr+roi_pos[1]]
            #sr_pos[0] = l[0]-rr+roi_pos[0]
            #sr_pos[1] = l[1]-rr+roi_pos[1]

            phot = absolute_intensity(roi, roi_pos)
            #phot = photometry_intensity(roi)

            pos_sr[peak][0] = sr_pos[0] # ypos
            pos_sr[peak][1] = sr_pos[1] # xpos
            pos_sr[peak][3] = phot # brightness


        print(f"Skipped {skip_counter} / {pos_candidate.shape[0]} ({skip_counter*100/pos_candidate.shape[0] :.2f}%) peaks in fitting due to irregular shape.")

        return pos_sr


#     def track_locs_in_z_all(self, pos_sr):
#        # assuming there is only one or less corresponding loc per frame in z
#        beads = {}
#        bidx = 0 # bead_idx 
#        run = True
#        z_loop = 0
#        max_frames = self.pp['planes']
#        #pos_sr = self.pos_sr.copy()
#        while run:
#            
#            # search for new bead position in current z_loop 
#            loc = next((l for l in pos_sr if l[2] == z_loop), None)
#            if loc is None:
#                if z_loop != max_frames:
#                    z_loop += 1
#                else:     
#                    run = False
#                    break
#
#            else:
#                # if unaccounted localisation exist, assign to a new bead and iterate through the stack to find its corresponding z-positions
#                beads[bidx] = [loc]
#                pos_sr = delete_loc(pos_sr, loc) 
#                #np.delete(pos_sr, loc, axis=0)
#                
#                for z in range(z_loop+1, max_frames):
#                    next_frame = pos_sr[pos_sr[:,2]==z]
#                    next_point = self.find_closest_neighbour_in_next_frame(beads[bidx][-1][:2], next_frame)
#                    if next_point is None:
#                        #beads[bidx].append(np.zeros(4))
#                        beads[bidx].append(['None', 'None', z, 'None'])
#                    else:
#                        beads[bidx].append(next_point)
#                        # remove loc from original loc array 
#                        pos_sr = delete_loc(pos_sr, next_point)
#
#                bidx += 1 # update bead idx after loop through stack for new assigment 
#        
#        return beads
#    
#
    def track_locs_in_z(self, pos_sr, bead_pos):
        # assuming there is only one or less corresponding loc per frame in z
        tracks = {}
        max_frames = self.pp['stack_height']
        #pos_sr = self.pos_sr.copy()
        for p, _ in enumerate(bead_pos): 
            tracks[p] = []
            for z in range(max_frames-1):
                next_frame = pos_sr[pos_sr[:,2]==z]
                if next_frame.any():
                    if z == 0 or not tracks[p]:
                        next_point = self.find_closest_neighbour_in_next_frame(bead_pos[p][:2], next_frame)
                    else:  
                        next_point = self.find_closest_neighbour_in_next_frame(tracks[p][-1][:2], next_frame)
                else: 
                    next_point = None

                if next_point is not None:
                    tracks[p].append(next_point)
                    # remove loc from original loc array 
                    pos_sr = delete_loc(pos_sr, next_point)    

            tracks[p] = np.array(tracks[p])
        return tracks


    def clean_up_tracks(self, tracks):
        # remove tracks are not long enough, defined by pp parameter
        # interpolate values that havent been detected from localisation or tracking step  

        # remove all traces that are below length threshold 
        cleaned_tracks = {}
        for k in tracks.keys():
            if len(tracks[k]) >= self.pp['frame_min']:
                # interpolate missing z values
                cleaned_tracks[k] = self.expand_and_interpolate(tracks[k], self.pp['stack_height'])
                #cleaned_tracks[k] = tracks[k]
        return cleaned_tracks

        
       


    def expand_and_interpolate(self, arr, target_length):
        n, cols = arr.shape
        assert cols == 4, "Input array must have shape (n, 4)"
        
        if n >= target_length:
            return arr
        
        # Create a full z array from min to max + 1 with step 1
        min_z = 0
        max_z = self.pp['stack_height']
        full_z = np.arange(min_z, max_z)
        
        # Create a new array with the full set of z values
        new_arr = np.zeros((target_length, 4))
        
        # Set the z positions
        new_arr[:,3] = full_z

        arr_keys = np.delete(range(cols), 2)

        # Interpolate x, y, and the value at the z position
        for i in arr_keys:
            new_arr[:, i] = np.interp(full_z, arr[:, 2], arr[:, i])
        
        
        return new_arr



    def find_earliest_neighbour_in_next_frame(self, p, arr):
        # go through every element, calculate distance to current point
        # if point is within distnace, stop iteration 
        # return the index in the list 
        # p: (x1,y1) of target point
        # arr: np.array of neighbouring candidates
        
        for i, point in enumerate(arr):
            if self.get_distance(p, point) <= self.pp['d_max']:
                return point
            else:
                return None  # if no point is within  distance
            


    def find_closest_neighbour_in_next_frame(self, p, arr):
        # return the index in the list 
        # p: (x1,y1) of target point
        # arr: np.array of neighbouring candidates
        
        distances = np.sum((arr[:,:2] - np.array(p))**2, axis=1)
        
        # Find the index of the minimum distance
        closest_index = np.argmin(distances)

        if distances[closest_index] <= self.pp['d_max']**2:
            return arr[closest_index]
        else: 
            return None


    def get_distance(self, a, b):
        # get eucledian distance betwwen point a & b
        # a : (x1, y1)
        # b: (x2, y2)
        d = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5

        return d


    def get_dz(self):
        x = range(self.pp['stack_height'])
        res = []
        fits = []
        data = []
        n = []
        for t in self.tracks.keys():
            y=np.mean(self.tracks[t][:,:,3], axis=0)
            # Fit the Gaussian to the data
            data.append(y)
            params = self.fit_gaussian(x, y)
            res.append(params)
            fits.append(gaussian(x, *params))
            n.append(self.tracks[t].shape[0])

        res = np.array(res)
        data = np.array(data)
        fits = np.array(fits)
        new_order = np.argsort(res[:,1])

        f = plot2LinesVerticalMarkers(data, fits, n, res[:,1],  "grayvalue [1/units]", self.pp['zstep'])
        self.figs['dz'] = f

        res = res[new_order, :] 
        self.order = new_order
        self.dz['dz'] = [(res[i+1,1]-res[i,1])*self.pp['zstep'] for i in range(res.shape[0]-1)]
        self.dz['labels'] = [f'{i+1}-{i}' for i in range(res.shape[0]-1)]
        self.dz['fp'] = [int(nz) for nz in res[:,1]]

        return self.dz, self.order
    

    def phasor_localise(self, roi):
        #Perform 2D Fourier transform over the complete ROI
        roi_f = np.fft.fft2(roi)
        xangle = np.arctan(roi_f[0,1].imag/roi_f[0,1].real) - np.pi
        #Correct in case it's positive
        if xangle > 0:
            xangle -= 2*np.pi
        #Calculate position based on the ROI radius
        xpos = abs(xangle)/(2*np.pi/(self.pp['roi']*2+1))+0.5

        #Do the same for the Y angle and position
        yangle = np.arctan(roi_f[1,0].imag/roi_f[1,0].real) - np.pi
        if yangle > 0:
            yangle -= 2*np.pi
        ypos = abs(yangle)/(2*np.pi/(self.pp['roi']*2+1))+0.5

        return (xpos,ypos)
    
    def fit_gaussian(self, x, y):
        # Initial guesses for the parameters
        amp_init = np.max(y) - np.min(y)
        mean_init = np.sum(x * y) / np.sum(y)
        stddev_init = np.sqrt(np.sum(y * (x - mean_init) ** 2) / np.sum(y))
        offset_init = np.min(y)
        
        bounds_low = (0, 0, 0.1, 0) # amp, mean, stddev, offset
        bounds_up = (10*amp_init, np.max(x), 100*stddev_init, np.max(y)) # amp, mean, stddev, offset

        # Use curve_fit to fit the Gaussian function to the data
        popt, pcov = curve_fit(gaussian, x, y, p0=[amp_init, mean_init, stddev_init, offset_init], bounds=(bounds_low, bounds_up))

        return popt

    def expand_positions_in_z(self, pos):
        #pos: list of x,y positions of potential beads
        # expand the list for all positions in z to fit into regular processing
        h = self.pp['stack_height']
        zpos = np.array(range(h))
        for p in range(pos.shape[0]):
            temp=np.empty((h, 3))
            temp[:,0] = pos[p][0]
            temp[:,1] = pos[p][1]
            temp[:,2] = zpos

            if p == 0: 
                pos_o = temp
            else: 
                pos_o = np.concatenate((pos_o, temp), axis=0)

        return pos_o.astype(int)
    


#     def get_transformation(self, stack, refplane):
#        # determine affine transformation between planes
#        planes = self.pp['planes']
#        self.refplane = refplane
#        
#        # if markers are empty, find them first
#        outer = tqdm(total=planes, desc='Finding markers', position=0)
#
#        for p in range(planes):
#            self.markers[p]= self.find_markers(stack[p,...], self.beadID[p], p)
#            outer.update(1)
#
#        # calculate tranformation
#        outer = tqdm(total=planes, desc='Calculating transform')
#        transform_error = {}
#        for p in range(planes):
#            self.transform[p], self.markers[p], transform_error[p] = self.calculate_transform(ref=self.markers[refplane], tar=self.markers[p])
#            outer.update(1)
#
#            self.transform_error[p] = {}
#            self.transform_error[p]['mean'] = np.mean(transform_error[p])
#            self.transform_error[p]['std'] = np.std(transform_error[p])
#            self.transform_error[p]['marker_count'] = len(transform_error[p])
#            print(f"Plane {p}: Pixel2pixel_error: {self.transform_error[p]['mean']:.4f} +-: {self.transform_error[p]['std']:.4f}, with Pixel2pixel_error: {self.transform_error[p]['marker_count']} markers")
#
#        return self.transform, self.transform_error, self.markers
#
    def match_markers(self, ref, tar):
        # match keypoints of target plane to reference plane
        #matches = feature.match_descriptors(ref, tar, max_distance=20, cross_check=True) 
        matches = feature.match_descriptors(ref, tar, cross_check=True) 
        ref_match = ref[matches[:,0]]
        tar_match = tar[matches[:,1]]
        return ref_match, tar_match

    def apply_transformation(self, stack, transform):
        assert len(stack.shape) == 4, "Input stack must have 4 dimensions"
        #assert stack.shape[0]-1 == len(transform.keys()), f"Not enough transformations ({len(transform.keys())}) to apply to stack with {stack.shape[0]} planes"
        
        if 'planes' in self.pp.keys():
            planes = self.pp['planes']
        else:
            planes = stack.shape[0]
            
        h = stack.shape[1]

        # apply tranformation iteratively for every slice and plane
        outer = tqdm(total=planes-1, desc='Applying transform', position=0)
        for p, pt in zip(range(1,planes),transform.values()):
            inner = tqdm(total=h, desc='Slice', position=0)
                        # convert json ecnoded list to numpy array if needed
            if isinstance(pt, list):
                pt = np.array(pt)
            for t in range(h):
                # Transform a single image using the affine transformation matrix
                #stack[p,t,...] = ndimage.affine_transform(stack[p,t,...], pt[:, :2], offset=pt[:, 2])
                stack[p,t,...] = ndimage.affine_transform(np.squeeze(stack[p,t,...]), pt[:2, :2], offset=pt[:2, 2])
                inner.update(1)
                
            outer.update(1)

        return stack
    
    '''
    def calculate_transform(self, ref, tar):
        # ensure ref & tar are of equal length and shorten if not
        pretranslate = self.pretranslate
        if pretranslate:            
            if ref.shape > tar.shape:
                ref = self.remove_outliers(ref)
            elif ref.shape < tar.shape:
                tar = self.remove_outliers(tar)

            ref, tar, tt = self.pre_translate_markers(ref, tar) # tt =(tx,ty) shifts 

        ref_match, tar_match = self.match_markers(ref, tar)

        affine_matrix, mask = cv2.estimateAffine2D(ref_match, tar_match)
        if pretranslate:
            affine_matrix[0,2] += tt[0]
            affine_matrix[1,2] += tt[1]

        

        tar_match = self.apply_affine_transform_2points(tar_match, affine_matrix)
        plt.figure()
        plt.scatter(tar[:,0], tar[:,1], color='r', marker="o", alpha=0.3)
        plt.scatter(ref[:,0], ref[:,1], color="black", marker="o", alpha=0.3)

        plt.scatter(tar_match[:,0], tar_match[:,1], color='r', marker='+')
        plt.scatter(ref_match[:,0], ref_match[:,1], color="black", marker='+')
        plt.legend(["target", "reference", "warped target", "matched reference"])
        plt.show()

        error_xy = ref_match-tar_match
        error_all = [np.sqrt((cc[0]**2) + (cc[1]**2)) for cc in error_xy]
        
        return affine_matrix, tar_match, error_all
    
    '''

    def apply_affine_transform_2points(self, points, transform_matrix):
        """
        Applies an affine transformation to a set of 2D points using a 2x3 transformation matrix.
        
        Parameters:
        - points (np.ndarray): An Nx2 array of 2D points (each row is a point [x, y]).
        - transform_matrix (np.ndarray): A 2x3 affine transformation matrix.
        
        Returns:
        - np.ndarray: The transformed Nx2 array of points.
        """
        # Convert points to homogeneous coordinates by adding a column of ones
        num_points = points.shape[0]
        homogeneous_points = np.hstack([points, np.ones((num_points, 1))])
        
        # Apply the affine transformation
        transformed_points = homogeneous_points @ transform_matrix.T
        
        return transformed_points
    
#     def remove_outliers(self, data, lower_percentile=5, upper_percentile=95):
#        """
#        Removes outliers from a 2D array using percentile-based filtering.
#        
#        Parameters:
#        - data (np.ndarray): A 2D numpy array of numerical values.
#        - lower_percentile (float): The lower percentile threshold. Values below this percentile will be considered outliers.
#        - upper_percentile (float): The upper percentile threshold. Values above this percentile will be considered outliers.
#        
#        Returns:
#        - np.ndarray: A 2D numpy array with outliers removed.
#        """
#        mask = np.ones(data.shape[0], dtype=bool)
#        for i in range(data.shape[1]):
#            # Calculate the specified percentiles
#            lower_bound = np.percentile(data[:,i], lower_percentile)
#            upper_bound = np.percentile(data[:,i], upper_percentile)
#            
#            # Create a mask for values within the bounds
#            mask = mask & (data[:,i] >= lower_bound) & (data[:,i] <= upper_bound)
#        # Set outliers to NaN or remove them (depending on your needs)
#        cleaned_data = data[mask,:]  # Use np.nan for missing data representation
#        return cleaned_data  # Returning both for flexibility
#
#
    def find_markers(self, stack, id, plane=None):
        # localise candidates from MIP stack
        assert len(stack.shape)==3, "stack has wrong dimensions"
        assert plane != None, "provide plane information to use focal plane for marker ID, now using maximum projection"
        z_pos, Nx, Ny = stack.shape

        try: 
            mip = stack[self.dz['fp'][plane], ...]
            id = self.locs_from_2d(mip) 
        except:
            print("reverting to maximum projection in marker ID")
            mip = np.max(stack, axis=0)
        
        rr = self.pp['roi']
        pos_sr = np.empty((id.shape[0], 2), dtype=np.float32)
        for peak in range(id.shape[0]):
            l = id[peak]
            roi = mip[int(l[0]-rr):int(l[0]+rr), int(l[1]-rr):int(l[1]+rr)]
            
            try:
                # sr position in roi crop 
                roi_pos = list(self.phasor_localise(roi))

                # update to global coordinates
                sr_pos = [l[0]-rr+roi_pos[0], l[1]-rr+roi_pos[1]]

                pos_sr[peak][0] = sr_pos[0] # ypos
                pos_sr[peak][1] = sr_pos[1] # xpos
            except:
                print(f"Skipping peak {peak} in marker ID process")

        return pos_sr
    

#     def display_transformations(self):
#        # Create a simple grid
#        x = np.linspace(0, 100, 26)
#        y = np.linspace(0, 100, 26)
#        grid_x, grid_y = np.meshgrid(x, y)
#        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
#        
#        # Plot the original and transformed grids
#        fig, axs = plt.subplots(1, len(self.transform.keys())+1, figsize=(15, 10))
#
#        plot_grid(axs[0], grid_points, 'default')
#
#        for ax, (name, matrix) in zip(axs[1:], self.transform.items()):
#            transformed_points = apply_display_affine_transform(matrix, grid_points)
#            plot_grid(ax, transformed_points, name)
#
#        plt.tight_layout()
#        plt.show()
#
#
    '''
    def pre_translate_markers(self, ref, tar):
        # Calculate the centroids of ref and tar
        centroid_ref = np.mean(ref, axis=0)
        centroid_tar = np.mean(tar, axis=0)
        
    
        plt.figure()
        plt.scatter(tar[:,0], tar[:,1], color='r')
        plt.scatter(ref[:,0], ref[:,1])
        plt.scatter(centroid_ref[0], centroid_ref[1], color="black")
        plt.scatter(centroid_tar[0], centroid_tar[1], color="green")

        # Step 2: Calculate the translation vector
        tt = centroid_ref - centroid_tar
        
        # Step 3: Apply the translation to tar
        aligned_tar = tar + tt
        plt.scatter(aligned_tar[:,0], aligned_tar[:,1], color='r', marker='+')
        plt.show()

        return ref, aligned_tar, tt
    


#     def estimate_dxy(self, tracks):
#        # estimate the displacement along z per bead in a dict of beads
#        # tracks: dict(bead_1, bead_2, bead_3)
#        # bead_n: array([x1,y1,z1,brightness1;
#        #                x2, y2, z2, brightness2])
#        dxy = {}
#        for k in tracks.shape[0]:
#   
#            # interpolate missing z values
#            dxy[k][0] = tracks[k][:,0]
#            #cleaned_tracks[k] = tracks[k]
#        return
#
    '''
#############################################
# new quad based method, just sample all quad based transforms, then take the best
    def get_affine_transform_via_quads(self, stack, refplane):

        if self.log:
            print('Using quad based transformation estimation, whole quad combination sampling')
        # determine affine transformation between planes
        planes = self.pp['planes']
        
        # if markers are empty, find them first
        outer = tqdm(total=planes, desc='Finding markers', position=0)
        for p in range(planes):
            self.markers[p]= self.find_markers(stack[p,...], self.beadID[p], p)
            outer.update(1)

        outer = tqdm(total=planes, desc='Query quads, estimate transform')
        transform_error_estimate = {}
        for p in range(planes):
            transform_error_estimate[p], self.transform[p], self.quad_match[p] = self.estimate_affine_via_quads(self.markers[refplane], self.markers[p], k=4)
            outer.update(1)




        '''
        # create tree and quad from loc positions
        for p in range(planes):
            [kd, quad] = makeTreeAndQuads(x = self.markers[p][:,0], 
                                            y = self.markers[p][:,1],
                                            min_size = self.pp['min_size'],
                                            max_size = self.pp['max_size'],
                                            max_neighbors = self.pp['max_neighbours'])
            self.kd[p] = kd
            self.quad[p] = quad
            outer.update(1)

        # calculate tranformation
        outer = tqdm(total=planes, desc='Calculating transform')
        transform_error_estimate = {}
        for p in range(planes):
            [transform_error_estimate[p], self.transform[p], self.quad_match[p]] = self.findTransform(ref_quad=self.quad[refplane],
                                                                                    other_quad = self.quad[p],
                                                                                    ref_kd=self.kd[refplane],
                                                                                    other_kd=self.kd[p])
            outer.update(1)

            if (transform_error_estimate[p] > 10.0):
                plotMatch(self.kd[refplane],
                        self.kd[p],
                        self.transform[p],
                        #save_as = f"transform_ref{refplane}_other{p}.png",
                        show = self.log)
        '''
        if self.check_magnification:
            outer = tqdm(total=planes, desc='Creating quads')
            # create tree and quad from loc positions
            for p in range(planes):
                if self.quad_match[p]['ref'] is None:
                    self.scaling_factor[p] = None
                    print(f'Skipping magnification check for plane {p}, no valid points found.')
                else:
                    self.scaling_factor[p] = self.compute_scaling_factor(self.quad_match[p]['ref'], 
                                                            self.quad_match[p]['other'] )
                
                
            
        return self.transform, self.transform_error, self.markers, self.scaling_factor



    def estimate_affine_via_quads(self, src, dst, k=4):
        src_tree = cKDTree(src)
        dst_tree = cKDTree(dst)
        idx_src = src_tree.query(src, k=k)[1]
        idx_dst = dst_tree.query(dst, k=k)[1]
        
        best_error = float('inf')
        best_tf = None

        for i in range(len(src)):
            quad_src = src[idx_src[i]]
            quad_dst = dst[idx_dst[i]]
            if len(quad_src) >= 3 and len(quad_dst) >= 3:
                try:
                    tf = estimate_transform('affine', quad_src, quad_dst)
                    transformed = tf(quad_src)
                    error = np.mean(np.linalg.norm(transformed - quad_dst, axis=1))
                    if error < best_error:
                        best_error = error
                        best_tf = tf
                        best_quad_index = i
                except Exception:
                    continue

        return best_error, best_tf.params[:2, :], {'ref' : src[idx_src[best_quad_index]], 'other' :dst[idx_dst[best_quad_index]]}



# end of new quad sampling method
######################################################


#     def get_micrometry_transformation(self, stack, refplane):
#        # calculate affine transform via babcock code base  
#        if self.log:
#            print('Using babcock quad based transformation estimation')
#        # determine affine transformation between planes
#        planes = self.pp['planes']
#        
#        # if markers are empty, find them first
#        outer = tqdm(total=planes, desc='Finding markers', position=0)
#        for p in range(planes):
#            self.markers[p]= self.find_markers(stack[p,...], self.beadID[p], p)
#            outer.update(1)
#
#        outer = tqdm(total=planes, desc='Creating quads')
#        # create tree and quad from loc positions
#        for p in range(planes):
#            [kd, quad] = makeTreeAndQuads(x = self.markers[p][:,0], 
#                                            y = self.markers[p][:,1],
#                                            min_size = self.pp['min_size'],
#                                            max_size = self.pp['max_size'],
#                                            max_neighbors = self.pp['max_neighbours'])
#            self.kd[p] = kd
#            self.quad[p] = quad
#            outer.update(1)
#
#        # calculate tranformation
#        outer = tqdm(total=planes, desc='Calculating transform')
#        transform_error_estimate = {}
#        for p in range(planes):
#            [transform_error_estimate[p], self.transform[p], self.quad_match[p]] = self.findTransform(ref_quad=self.quad[refplane],
#                                                                                    other_quad = self.quad[p],
#                                                                                    ref_kd=self.kd[refplane],
#                                                                                    other_kd=self.kd[p])
#            outer.update(1)
#
#            if (transform_error_estimate[p] > 10.0):
#                plotMatch(self.kd[refplane],
#                        self.kd[p],
#                        self.transform[p],
#                        #save_as = f"transform_ref{refplane}_other{p}.png",
#                        show = self.log)
#
#        if self.check_magnification:
#            outer = tqdm(total=planes, desc='Creating quads')
#            # create tree and quad from loc positions
#            for p in range(planes):
#                if self.quad_match[p]['ref'] is None:
#                    self.scaling_factor[p] = None
#                    print(f'Skipping magnification check for plane {p}, no valid points found.')
#                else:
#                    self.scaling_factor[p] = self.compute_scaling_factor(self.quad_match[p]['ref'], 
#                                                            self.quad_match[p]['other'] )
#                
#                
#            
#        return self.transform, self.transform_error, self.markers, self.scaling_factor
#    
    def compute_scaling_factor(self, points_ref, points_other):
        '''
        If image has a different magnification, relative distances between markers should remain consistent
        but the absolute distances will scale by a magnification factor.
        Should be equal amount of markers -> use quads from transformation estimation. 
        '''
        #dsitance matrices
        D_ref = self.compute_distance_matrix(points_ref)
        D_other = self.compute_distance_matrix(points_other)

        # normalise
        D_ref_norm = self.normalize_matrix(D_ref)
        D_other_norm = self.normalize_matrix(D_other)

        #scaling fator = 
        sfs = D_other_norm/D_ref_norm
        sf = np.median(sfs[np.isfinite(sfs)])

        return sf


    def compute_distance_matrix(self, points):
       n = points.shape[0]
       dist_matrix = np.zeros((n, n))
       for i in range(n):
           for j in range(n):
               dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])
       return dist_matrix

    def normalize_matrix(self, matrix):
       return matrix / np.max(matrix)

################################################################################
# Micrometry babcock
################################################################################

    def findTransform(self, ref_quad, other_quad, ref_kd, other_kd, tolerance = None, min_size = None, max_size = None, max_neighbors = None):

        if max_neighbors is None:
            max_neighbors = self.pp['max_neighbours']
            
        if max_size is None:
            max_size = self.pp['max_size']
            
        if min_size is None:
            min_size = self.pp['min_size']

        if 'density' not in self.pp.keys():
            if 'stack_sx' not in self.pp.keys():
                print("Couldn't find stack dimensions, estimating loc density to 1/(30*30)um**2")
                self.pp['density'] = 1/(30**2)
            else:
                self.pp['density'] = 1/(self.pp['stack_sx']*self.pp['stack_sy'])
        
        if tolerance is None:
            tolerance = self.pp['tolerance']

        if self.log:
            print("")
            print("Comparing quads.")
        
        #
        # Unlike astrometry.net we are just comparing all the quads looking for the
        # one that has the best score. This should be at least 10.0 as, based on
        # testing, you can sometimes get scores as high as 9.7 even if the match
        # is not actually any good.
        #
        best_ratio = 0.0
        best_transform = None
        matches = 0
        quad_match = {'ref': None, 
                        'other': None}
        for q1 in ref_quad:
            for q2 in other_quad:
                if q1.isMatch(q2, tolerance = tolerance):
                    fg_p = fgProbability(ref_kd, other_kd, q1.getTransform(q2), self.pp['density'])
                    ratio = math.log(fg_p/self.pp['density'])
                    if self.log:
                        print("Match {0:d} {1:.2f} {2:.2e} {3:.2f}".format(matches, fg_p, self.pp['density'], ratio))
                    if (ratio > best_ratio):
                        best_ratio = ratio
                        best_transform = q1.getTransform(q2) + q2.getTransform(q1)
                        quad_match['ref'] = np.vstack([q1.A, q1.B, q1.C, q1.D])
                        quad_match['other'] = np.vstack([q2.A, q2.B, q2.C, q2.D])
                    matches += 1

        if self.log:
            print("Found", matches, "matching quads")

        return [best_ratio, best_transform, quad_match]
    


###############################################################
# utilities
###############################################################

def applyTransform(kd, transform):
    tx = transform[0]
    ty = transform[1]
    x = tx[0] + tx[1]*kd.data[:,0] + tx[2]*kd.data[:,1]
    y = ty[0] + ty[1]*kd.data[:,0] + ty[2]*kd.data[:,1]
    return [x, y]


# def fgProbability(kd1, kd2, transform, bg_p):
#    """
#    Returns an estimate of how likely the transform is correct.
#    """
#    # Transform 'other' coordinates into the 'reference' frame.
#    [x2, y2] = applyTransform(kd2, transform)
#    p2 = np.stack((x2, y2), axis = -1)
#
#    # Calculate distance to nearest point in 'reference'.
#    [dist, index] = kd1.query(p2)
#
#    # Score assuming a localization accuracy of 1 pixel.
#    fg_p = bg_p + (1.0 - bg_p) * np.sum(np.exp(-dist*dist*0.5))/float(x2.size)
#    return fg_p
#
#    
def makeTreeAndQuads(x, y, min_size = None, max_size = None, max_neighbors = None):
    """
    Make a KD tree and a list of quads from x, y points.
    """
    kd = spatial.KDTree(np.stack((x, y), axis = -1))
    m_quads = quads.makeQuads(kd,
                              min_size = min_size,
                              max_size = max_size,
                              max_neighbors = max_neighbors)
    return [kd, m_quads]


# def plotMatch(kd1, kd2, transform, save_as = None, show = True):
#    [x2, y2] = applyTransform(kd2, transform)
#    
#    fig = plt.figure()
#    plt.scatter(kd1.data[:,0], kd1.data[:,1], facecolors = 'none', edgecolors = 'red', s = 100)
#    plt.scatter(x2, y2, color = 'green', marker = '+', s = 100)
#
#    legend = plt.legend(('reference', 'other'), loc=1)
#    plt.xlabel("pixels")
#    plt.ylabel("pixels")
#
#    ax = plt.gca()
#    ax.set_aspect('equal')
#
#    if save_as is not None:
#        fig.savefig(save_as)
#    
#    if show:
#        plt.show()
#
#
# def prettyPrintTransform(transform):
#    """
#    Pretty print the transform.
#    """
#    print("Reference to other transform:")
#    print("  {0:.4f} {1:.4f} {2:.4f}".format(transform[2][0], transform[2][1], transform[2][2]))
#    print("  {0:.4f} {1:.4f} {2:.4f}".format(transform[3][0], transform[3][1], transform[3][2]))
#    print("")
#    print("Other to reference transform:")
#    print("  {0:.4f} {1:.4f} {2:.4f}".format(transform[0][0], transform[0][1], transform[0][2]))
#    print("  {0:.4f} {1:.4f} {2:.4f}".format(transform[1][0], transform[1][1], transform[1][2]))
#    print("")
#
################################################################################
# END CLASS
################################################################################

def plot2LinesVerticalMarkers(data1, data2, n, xMarker, yval, ZDIST = 1):
    
    fig, ax = plt.subplots(figsize=(12,7))
    n_planes = len(data1[0])
    #cmap = cm.get_cmap('Pastel1', 8)
    z = np.linspace(0,ZDIST*n_planes, n_planes)
    idx = 0
    lgd = []
    LINES = []
    

    maxVal, minVal = np.max(data1), np.min(data1)
    
    for idx, (line, line2) in enumerate(zip(data1, data2)):

        LINES += ax.plot(z, line, color=clist[idx], alpha = 0.8, linewidth = 1)
        lgd.append("Ch{} - z: {:.0f}nm; A: {:.1f}, n: {}".format(idx, xMarker[idx]*ZDIST, np.max(line2), n[idx]))
        plt.plot(z, line2, color=clist[idx], linewidth = 1, linestyle='--', alpha = 1)
        plt.vlines(xMarker[idx]*ZDIST, minVal, maxVal, color=clist[idx])

    plt.ylim((minVal, maxVal))
    plt.xlim((0,np.max(z)))
    plt.xlabel("z 1/nm")
    plt.ylabel(yval)
    #leg = Legend(ax, LINES, lgd, loc='lower right', bbox_to_anchor=(0.5, 1.), frameon=False)
    leg = Legend(ax, LINES, lgd, loc='upper right')
    
    ax.add_artist(leg)
    
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    #fig = ax.get_figure()
    return fig, ax
    
#    output_name = os.path.join(outpath, fileSpecifier+".svg")
#    plt.savefig(output_name, dpi = 600, bbox_inches="tight", pad_inches=0.1, transparent=True)    
    

def absolute_intensity(ROI, xy):
    w = [xy[0]-np.floor(xy[0]), xy[1]-np.floor(xy[1])]

    c = np.array([[np.floor(xy[0]), np.floor(xy[0])+2], 
            [np.floor(xy[1]), np.floor(xy[1])+2]]).astype(np.uint8)# point coordinates 
    
    # check whether coords are at the border and shift if necessary
    if c[0][1] >= ROI.shape[0]-1:
        c[0] = c[0]-1

    if c[1][1] >= ROI.shape[1]-1:
        c[1] = c[1]-1
    val = approx_interp2(ROI[c[0][0]:c[0][1], c[1][0]:c[1][1]], w)

    #val = np.max(ROI)

    return val



def approx_interp2(val,w):
    #val: 2,2 neighbouring values
    #w: weights for interpolation 
    return (val[0,0]*w[0]+ val[1,0]*(1-w[0]))*w[1] + (val[0,1]*w[0]+ val[1,1]*(1-w[0]))*(1-w[1])

#################################################################
#The intensity-measure we will use is a photometry-based method, described in:
#S. Preus, L.L. Hildebrandt, and V. Birkedal, Biophys. J. 111, 1278 (2016).
#Also see Figure S11 in doi.org/10.1063/1.5005899 (Martens et al., 2017)

# def photometry_intensity(ROI):
#  #First we create emtpy signal and background maps with the same shape as
#  #the ROI.
#  SignalMap = np.zeros(ROI.shape)
#  BackgroundMap = np.zeros(ROI.shape)
#  #Next we determine the ROI radius from the data
#  ROIradius = (ROI.shape[0]-1)/2
#
#  #Now we attribute every pixel in the signal and background maps to be
#  #belonging either to signal or background based on the distance to the
#  #center
#  #For this, we loop over the x and y positions
#  for xx in range(0,ROI.shape[0]):
#    for yy in range(0,ROI.shape[1]):
#      #Now we calculate Pythagoras' distance from this pixel to the center
#      distToCenter = np.sqrt((xx-ROI.shape[0]/2+.5)**2 + (yy-ROI.shape[1]/2+.5)**2)
#      #And we populate either SignalMap or BackgroundMap based on this distance
#      if distToCenter <= (ROIradius): #This is signal for sure
#        SignalMap[xx,yy] = 1
#      elif distToCenter > (ROIradius-0.5): #This is background
#        BackgroundMap[xx,yy] = 1
#
#  #Now we take the 56th percentile of the data in the background map.
#  #This is a valid measure for the expected background intensity
#
#  #First we use the BackgroundMap as a mask for the intensity data, and use
#  #that to get a list of Background Intensities.
#  BackgroundIntensityList = np.ma.masked_array(ROI, mask=BackgroundMap).flatten()
#  #And then we take the 56th percentile (or the value closest to it)
#  if len(BackgroundIntensityList) > 0:
#    BackgroundIntensity = np.percentile(BackgroundIntensityList,56)
#  else:
#    BackgroundIntensity = 0
#
#  #To then assess the intensity, we simply sum the entire ROI in the SignalMap
#  #and subtract the BackgroundIntensity for every pixel
#  SignalIntensity = sum((ROI*SignalMap).flatten())
#  SignalIntensity -= BackgroundIntensity*sum(SignalMap.flatten())
#
#  #And we let the function return the SignalIntensity
#  return max(0,SignalIntensity)
#
#
#
def delete_loc(matrix, loc_to_delete):
    # Find the index of the row to delete
    index = np.where((matrix == loc_to_delete).all(axis=1))[0][0]
    # Delete the row using np.delete
    matrix = np.delete(matrix, index, axis=0)
    return matrix
#
#
def gaussian(x, amp, mean, stddev, offset):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + offset

# Function to plot the original and transformed grid
# def plot_grid(ax, points, title, color='b'):
#    #ax.plot(points[:, 0], points[:, 1], color + '-o', markersize=3)
#    ax.scatter(points[:, 0], points[:, 1], s=3, c=color, marker='o')
#    ax.set_title(title)
#    #ax.set_xlim(-2, 10)
#    #ax.set_ylim(-2, 10)
#    ax.grid(True)
#    ax.set_aspect('equal')
#
#
# def apply_display_affine_transform(matrix, points):
#    transformed_points = np.dot(matrix[:, :2], points.T).T + matrix[:, 2]
#    return transformed_points