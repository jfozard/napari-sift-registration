

import numpy as np
import numpy.random as npr

import pytest
from skimage.io import imread, imsave
from napari_sift_registration._widget import example_magic_widget

from skimage import data
from skimage.util import img_as_float
from skimage.transform import warp, AffineTransform
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage import segmentation

@pytest.fixture(scope="session")
def stacks_matching():

    npr.seed(1111)

    pos = npr.randint(200, size=(20,2))
    seeds = np.zeros((200,200), dtype=np.uint8)

    seeds[pos[:,0], pos[:,1]] = np.arange(1,21)

# See https://forum.image.sc/t/labeling-mitochondria-per-cell-basis-when-you-have-no-membrane-marker-and-only-nucleus/46636/6
    voronoi = segmentation.watershed(np.ones_like(seeds), seeds, compactness=1)

    img_orig = segmentation.find_boundaries(voronoi).astype(np.float64)
    

# warp synthetic image
    tform = AffineTransform(scale=(0.9, 0.9), rotation=0.2, translation=(20, -10))
    img_warped = warp(img_orig, tform.inverse, output_shape=(200, 200))

    img_orig = (255*(img_orig/np.max(img_orig))).astype(np.uint8)
    img_warped = (255*(img_warped/np.max(img_warped))).astype(np.uint8)
    
    yield img_orig, img_warped

# make_napari_viewer is a pytest fixture that returns a napari viewer object
## capsys is a pytest fixture that captures stdout and stderr output streams
def test_example_magic_widget(qtbot, make_napari_viewer, stacks_matching):
    
    viewer = make_napari_viewer()

    im1, im2 = stacks_matching
    
    layer = viewer.add_image(im1)
    layer = viewer.add_image(im2)

    my_widget = example_magic_widget()

    # if we "call" this object, it'll execute our function
    worker = my_widget(viewer, viewer.layers[0], viewer.layers[1])

    # Solution from napari-pystackreg
    with qtbot.waitSignal(
        worker.finished, timeout=30000
    ) as blocker:  
        pass
    
    
    assert(len(viewer.layers)==5)
