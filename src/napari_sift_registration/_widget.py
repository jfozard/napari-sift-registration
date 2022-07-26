
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
import napari
import napari.viewer

from skimage import transform
from skimage.feature import SIFT, match_descriptors

from skimage.measure import ransac
from skimage.transform import warp, AffineTransform

from napari.qt.threading import thread_worker

import numpy as np

#
@thread_worker(progress={"total":4})
def SIFT_registration(A, B, params):

    p_d = { k:v for k,v in params.items() if k in ['upscaling', 'n_octaves', 'n_scales', 'sigma_min', 'n_hist', 'n_ori']}
    descriptor_extractor = SIFT(**p_d)
    #descriptor_extractor = SIFT()

    descriptor_extractor.detect_and_extract(A)
    keypointsA = descriptor_extractor.keypoints
    descriptorsA = descriptor_extractor.descriptors


    descriptor_extractor.detect_and_extract(B)
    keypointsB = descriptor_extractor.keypoints
    descriptorsB = descriptor_extractor.descriptors

    yield
    
    matchesAB = match_descriptors(descriptorsA, descriptorsB, max_ratio=params['max_ratio'],
                              cross_check=True)

    matchesA, matchesB = keypointsA[matchesAB[:, 0]], keypointsB[matchesAB[:, 1]]

    model = AffineTransform()
    model.estimate(matchesA, matchesB)


    # Robustly estimate transform model with RANSAC
    transform_robust, inliers = ransac((matchesA, matchesB), AffineTransform, min_samples = params['min_samples'], residual_threshold = params['residual_threshold'], max_trials = params['max_trials'])
    
    outliers = inliers == False

    yield
    
    # Apply transformation to image
    warpA = np.ascontiguousarray(warp(A.T, transform_robust.inverse, order = 1, mode = "constant", cval = 0, clip = True, preserve_range = True).T)

    return keypointsA, keypointsB, matchesAB, inliers, warpA

@magic_factory(img_layer_moving={'label': 'Moving image layer', 'tooltip': 'Name of moving image'}, 
               img_layer_fixed={'label':'Fixed image layer', 'tooltip': 'Name of fixed image layer'},
               upsampling={'label':'Upsampling before feature detection',
                           'tooltip':'Upsampling before feature detection',
                            "choices": [1, 2, 4]},
               n_octaves = {'label': 'Maximum number of octaves',
                            'tooltip': 'Maximum number of octaves'},
               n_scales = {'label': 'Maximum number of scales in every octave',
                           'tooltip': 'Maximum number of scales in every octave'},
               sigma_min = {'label': 'Blur level of seed image',
                            'tooltip': 'Blur level of the seed image (after upsampling)' },
               n_hist = {'label': 'Feature descriptor size',
                         'tooltip': 'Number of bins in the histogram that describes the gradient orientations around keypoint'},
               n_ori = { 'label': 'Feature descriptor orientation bins',
                          'tooltip': 'Number of bins in the histograms of the descriptor patch'},
               max_ratio = { 'label': 'Closest/next closest ratio',
                          'tooltip': 'Maximum ratio of distances between first and second closest descriptor in the second set of descriptors' },
               min_samples = {'label': 'Minimum number of points sampled for each RANSAC model',
                              'tooltip': 'Minimum number of points sampled for each RANSAC model'},
               residual_threshold = {'label': 'Distance for points to be inliers in RANSAC model',
                                     'tooltip':  'Distance for points to be inliers in RANSAC model'},
               max_trials = {'label': 'Maximum number of trials in RANSAC model',
                             'tooltip': 'Maximum number of trials in RANSAC model'},
                          )
def example_magic_widget(viewer: 'napari.viewer.Viewer',
                         img_layer_moving: 'napari.layers.Layer',
                         img_layer_fixed: 'napari.layers.Layer',
                         upsampling:int=2,
                         n_octaves:int=8,
                         n_scales:int=3,
                         sigma_min:float=1.6,
                         n_hist:int=4,
                         n_ori:int=8,
                         max_ratio:float=0.7,
                         min_samples:int=5,
                         residual_threshold:float=2.0,
                         max_trials:int=1000,
                         ):
                         
    A = img_layer_moving.data
    B = img_layer_fixed.data

    params = {}
    params['upsampling'] = upsampling
    params['n_octaves'] = n_octaves
    params['n_scales'] = n_scales
    params['sigma_min'] = sigma_min
    params['n_hist'] = n_hist
    params['n_ori'] = n_ori
    params['max_ratio'] = max_ratio
    params['min_samples'] = min_samples
    params['residual_threshold'] = residual_threshold
    params['max_trials'] = max_trials


    
    def update_viewer(r):
        keypointsA, keypointsB, matchesAB, inliers, warpA = r
        print('update_viewer')
        
        moving_labels_base = np.cumsum(inliers)
        moving_labels = [ str(moving_labels_base[i]) if inliers[i] else '' for i in range(len(inliers)) ]
        viewer.add_points(keypointsA[matchesAB[:,0]],
                          face_color='inliers',
                          face_color_cycle=['magenta', 'green'],
                          properties={'inliers': inliers, 'label': moving_labels }, name='Keypoints Moving', text='{label}')

        fixed_labels_base = np.cumsum(inliers)
        fixed_labels = [ str(fixed_labels_base[i]) if inliers[i] else '' for i in range(len(inliers)) ]

        viewer.add_points(keypointsB[matchesAB[:,1]],
                          face_color='inliers',
                          face_color_cycle=['magenta', 'green'],
                          properties={'inliers': inliers, 'label':fixed_labels }, name='Keypoints Fixed', text='{label}')

        viewer.add_image(warpA, name='Warped Moving Image')
        print('done_update')
        
    worker = SIFT_registration(A, B, params)

    worker.returned.connect(update_viewer)
    worker.start()
    
    print('STARTED WORKER', worker, dir(worker), worker.abort_requested, worker.is_running)
    return worker

