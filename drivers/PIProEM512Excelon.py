import numpy as np
import time

# Import class for camera
from picam import PICam

# picam_types needed to convert settings to numerical values
from picam.picam_types import PicamReadoutControlMode, PicamAdcQuality, PicamAdcAnalogGain, PicamTriggerDetermination
from picam.picam_types import PicamTriggerResponse

# Error logging
import logging

class PIProEM512Excelon:
    def __init__(self, time_offset, exposure_time, EM_gain, analog_gain, CCD_temp, demo = False, ) -> None:
        # Time offset is offset from UNIX time 
        # (allows use of single precision floats in data storage)
        self.time_offset = time_offset

        # Store parameters
        self.exposure_time = exposure_time
        self.EM_gain = EM_gain
        self.analog_gain = analog_gain
        self.CCD_temp = CCD_temp
        self.demo = demo

        # Initialize camera class and load the Picam.dll library
        self.cam = PICam()
        self.cam.loadLibrary()

        #####################
        # Connect to camera #
        #####################
        try:
            # Find available cameras
            self.cam.getAvailableCameras(demo = demo)

            # Connect to camera ("camID = None" means connect to first camera found)
            self.cam.connect(camID = None)

        except AssertionError:
            logging.error("Error in connecting to PIProEM512Excelon")
            self.verification_string = "Can't connect to camera"
            self.instr = False
            return

        # Verify operation
        self.verification_string = "not implemented"

        #####################
        # Set camera params #
        #####################
        # Camera temperature
        self.cam.setParameter("SensorTemperatureSetPoint", float(CCD_temp))

        # Set exposure time
        self.cam.setParameter("ExposureTime", float(exposure_time)) # Exposure time in ms

        # Readout mode (FullFrame mean camera reads out one whole frame at a time)
        self.cam.setParameter("ReadoutControlMode", PicamReadoutControlMode["FullFrame"])

        # Set region of interest (x, width, x_binning, y, height, y_binning)
        # x = coordinate of first column, y = coordinate of first row
        self.cam.setROI(0, 512, 1, 0, 512, 1)

        # ADC parameters
        self.cam.setParameter("AdcQuality", PicamAdcQuality["ElectronMultiplied"])
        self.cam.setParameter("AdcAnalogGain", PicamAdcAnalogGain[analog_gain])
        self.cam.setParameter("AdcEMGain", int(EM_gain))
        self.cam.setParameter("AdcSpeed", 5.0)

        # sensor cleaning
        self.cam.setParameter("CleanSectionFinalHeightCount", 1)
        self.cam.setParameter("CleanSectionFinalHeight", 100)
        self.cam.setParameter("CleanSerialRegister", False)
        self.cam.setParameter("CleanCycleCount", 1)
        self.cam.setParameter("CleanCycleHeight", 100)
        self.cam.setParameter("CleanUntilTrigger", True)

        # Reaction to external trigger
        # if using a demo camera, don't want to wait for trigger
        if demo:
            self.cam.setParameter("TriggerResponse", PicamTriggerResponse["NoResponse"])
        
        else:            
            self.cam.setParameter("TriggerDetermination", PicamTriggerDetermination["RisingEdge"])
            self.cam.setParameter("TriggerResponse", PicamTriggerResponse["ReadoutPerTrigger"])

        # Apply settings
        self.cam.sendConfiguration()

        ######################
        # Output data params #
        ######################
        # Shape and type of output data
        self.shape = (1,512,512)
        self.dtype = np.int16

        # New attributes for HDF file
        self.new_attributes = []



    ###########################
    # Methods for python with #
    ###########################
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cam.disconnect()
        self.cam.unloadLibrary()


    ############################
    # Reading data from camera #
    ############################
    def ReadValue(self):
        # Read one frame from camera
        data = self.cam.readNFrames(N = 1, timeout = 6000)[0].reshape(self.shape)

        # If using a demo camera, add some random noise
        if self.demo:
            noise = np.random.rand(*data.shape)*data.mean()/3
            data += noise

        # Get timestamp
        timestamp = time.time()-self.time_offset

        # Make list of dictionaries that contain attributes for the frame
        all_attrs = []
        attrs = {"timestamp":timestamp, "exposure_time": self.exposure_time, "EM_gain": self.EM_gain,
                 "analog_gain":self.analog_gain, "CCD_temp": self.CCD_temp
                 }
        all_attrs.append(attrs)

        # Return the data and timestamp
        return [data, all_attrs]

    def GetWarnings(self):
        return None