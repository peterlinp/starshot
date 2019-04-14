#!/usr/bin/python
# $Id: synth-star-shot.py,v 1.23 2019/04/06 21:17:23 peterlin Exp peterlin $

import imp
import numpy as np
from datetime import datetime
# Razlicni verziji PyDicom se obnasata malo razlicno
try:
    imp.find_module('pydicom')
    #import pydicom, pydicom.UID
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    PyDicom1 = True
except:
    import dicom as pydicom
    #import dicom.UID as pydicom.UID
    from dicom.dataset import Dataset, FileDataset
    PyDicom1 = False

MyUID = '1.2.826.0.1.3680043.9.5318.'
# We create DICOM image files imgsize x imgsize
imgsize = 512

def lineprofile(x, y, amplitude = 255, baseline = 0, fwhm = 6, degree = 2,
                rotation = 0, offset = 0, dpi = 72):
    """
    Calculates a 1D (super-)Gaussian profile on a 2D array.

    Parameters
    ----------
    x, y : 2D meshgrids
    amplitude : height of the peak of the Gaussian above the baseline
    baseline : baseline value
    fwhm : full width of the line at half maximum (in millimeters)
    degree : degree of the super-Gaussian curve; 2 (default) means Gaussian
    rotation : the angle of the Gaussian profile line (in degrees;
               0 being 12 o'clock and increasing clockwise)
    offset : the displacement of the center of the line from the center of the
             image (in millimeters)
    dpi : image resolution (dots/inch; default 72)

    Returns
    -------
    2D meshgrid with the calculated values
    """
    theta = rotation*np.pi/180
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    return np.round(baseline + amplitude*np.exp(-(x*np.cos(theta) +
                    y*np.sin(theta) - offset)**degree/(2*sigma**degree)))

# Stolen from https://stackoverflow.com/questions/14350675/create-pydicom-file-from-numpy-array
def write_dicom(pixel_array,filename):
    """
    Writes a 2D NumPy array as a DICOM file.

    Parameters
    ----------
    pixel_array : 2D NumPy ndarray.  The array dimension is currently not being
                  checked - beware!
    filename: string containing the name of the output file.
    """

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
    file_meta.MediaStorageSOPInstanceUID = MyUID+datetime.now().strftime("%Y%m%d.%H%M%S.%f")
    file_meta.ImplementationClassUID = MyUID+datetime.now().strftime("%Y%m%d.%H%M%S.%f")
    ds = FileDataset(filename, {},file_meta = file_meta,preamble="\0"*128)
    ds.Modality = 'SYN'
    ds.ContentDate = datetime.now().strftime("%Y%m%d")
    ds.ContentTime = datetime.now().strftime("%H%M%S")
    ds.StudyInstanceUID =  MyUID+datetime.now().strftime("%Y%m%d.%H%M%S.%f")
    ds.SeriesInstanceUID = MyUID+datetime.now().strftime("%Y%m%d.%H%M%S.%f")
    ds.SOPInstanceUID =    MyUID+datetime.now().strftime("%Y%m%d.%H%M%S.%f")
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
    ds.SecondaryCaptureDeviceManufctur = 'Python 2.7.3'

    ## These are the necessary imaging components of the FileDataset object.
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    #ds.SmallestImagePixelValue = '\\x00\\x00'
    #ds.LargestImagePixelValue = '\\xff\\xff'
    # Pylinac needs PixelSpacing (0028,0030) and RTImageSID (3002,0026)
    ds.PixelSpacing = "%5.3f\\%5.3f" % (25.4/72, 25.4/72)

    # Pipspro reads this tag, but its interpretation is weird
    # (DPI 0.07 instead of 72)
    ds.NominalScannedPixelSpacing = ds.PixelSpacing
    
    # PipsPro does not make use of these two
    ds.ImagerPixelSpacing = ds.PixelSpacing
    ds.ImagePlanePixelSpacing = ds.PixelSpacing
    
    ds.RTImageSID = 1000
    ds.Columns = pixel_array.shape[0]
    ds.Rows = pixel_array.shape[1]
    if pixel_array.dtype != np.uint16:
        pixel_array = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array.tostring()

    ds.save_as(filename, write_like_original=False)
    return

def six_spike_starshot(radius, filename, size=1024, dpi=72, angle=0, fwhm=6,
                       degree = 2):
    """
    ***Obsolete: please use create_starshot() instead***
    Create a six-spike synthetic starshot image and save it as a DICOM image
    for benchmarking the starshot image. The spikes are 60 degrees apart.

    Parameters
    ----------
    radius : the radius of the incircle, in millimeters
    size : the size of the square image matrix (in pixels; default 1024)
    dpi : the resolution of the image (in dots per pinch; default 72)
    angle : with angle equal to 0, one pair of spikes extend in the vertical
            direction; angle sets the deviation of this spike pair from 
            the vertical (in degrees; default 0)
    fwhm : spike width (full width at half maximum, in milimeters)
    degree : degree of the super-Gaussian curve; 2 (default) means Gaussian
    filename : string containing the name of the output DICOM image file.
    """
    create_starshot(radius = radius, filename = filename, size = size,
                    dpi = dpi, beams = 3, angle = angle, fwhm = fwhm,
                    degree = degree)

def create_starshot(radius, filename, size = 1024, dpi = 72, beams = 3,
                    angle = 0, fwhm = 6, degree = 2):
    """
    Create a synthetic starshot image with n intersecting beams and save it 
    as a DICOM image for benchmarking the starshot image. The spikes are 
    180/n degrees apart.

    Parameters
    ----------
    radius : the radius of the incircle, in millimeters
    size : the size of the square image matrix (in pixels; default 1024)
    dpi : the resolution of the image (in dots per pinch; default 72)
    beams : the number of intersecting beams; must be odd (default 3;
            number of spikes is twice the number of beams)
    angle : with angle equal to 0, one pair of spikes extend in the vertical
            direction; angle sets the deviation of this spike pair from 
            the vertical (in degrees; default 0)
    fwhm : spike width (full width at half maximum, in milimeters)
    degree : degree of the super-Gaussian curve; 2 (default) means Gaussian
    filename : string containing the name of the output DICOM image file.
    """
    
    if np.mod(beams, 2) != 1:
        raise ValueError('Number of beams should be odd')
    if beams > 10:
        raise ValueError('Number of beams should be 9 or less')
    if beams < 3:
        raise ValueError('Number of beams should be at least 3')

    xdim = int(size)
    # 1024 x 1024, 72 dpi, v milimetrih 2.88 pix/mm
    xresol = dpi/25.4
    x = y = np.linspace(-xdim/2, xdim/2-1, xdim)/xresol
    xv, yv = np.meshgrid(x, y)
    zv = np.zeros([int(beams), xdim, xdim])

    angle_step = 360./beams
    beam_list = np.arange(0, 360, angle_step)
    for i in range(len(beam_list)):
        angle = beam_list[i]
        zv[i,:,:] = lineprofile(xv, yv, baseline=900, amplitude=-800,
                                offset=radius, rotation=0+angle,
                                fwhm=fwhm, degree=degree)
    zvs = zv.sum(0)
    im = zvs.astype(np.uint16)

    write_dicom(im, filename)

# Create a series of benchmark DICOM files
#radius_list = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
radius_list = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
#angle_list = np.array([0, 1])
angle_list = np.array([0])
fwhm_list = np.array([3, 6, 10])
beams_list = np.array([3, 5, 7, 9])

for beams in beams_list:
    for angle_vert in angle_list:
        for incircle_radius in radius_list:
            for fwhm in fwhm_list:
                beams_str = '-%d' % beams
                fwhm_str = '-%02d' % fwhm
                angle_vert_str = '-%01d' % angle_vert
                incircle_rad_str = str(incircle_radius).replace('.', '')
                fname = 'starshot-bench' + beams_str + fwhm_str + angle_vert_str + '-' + incircle_rad_str + '.dcm'
                print(fname)
                create_starshot(radius=incircle_radius, filename=fname,
                                size=imgsize, dpi=72, beams = beams,
                                angle=angle_vert, fwhm=fwhm, degree=4)
