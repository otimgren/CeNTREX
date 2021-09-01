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
    def __init__(self, time_offset) -> None:
        # Time offset is offset from UNIX time 
        # (allows use of single precision floats in data storage)
        self.time_offset = time_offset

        # Initialize camera class and load the Picam.dll library
        self.cam.loadLibrary()

        #####################
        # Connect to camera #
        #####################
        try:
            # Find available cameras
            self.cam.getAvailableCameras()

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
        self.cam.setParameter("SensorTemperatureSetPoint", -70)

        # Set exposure time
        self.cam.setParameter("ExposureTime", 10) # Exposure time in ms

        # Readout mode
        self.cam.setParameter("ReadoutControlMode", PicamReadoutControlMode["FullFrame"])

        # ADC parameters
        self.cam.setParameter("AdcQuality", PicamAdcQuality["ElectronMultiplied"])
        self.cam.setParameter("AdcAnalogGain", PicamAdcAnalogGain["Low"])
        self.cam.setParameter("AdcEMGain", 1)

        # sensor cleaning
        self.cam.setParameter("CleanSectionFinalHeightCount", 1)
        self.cam.setParameter("CleanSectionFinalHeight", 100)
        self.cam.setParameter("CleanSerialRegister", False)
        self.cam.setParameter("CleanCycleCount", 1)
        self.cam.setParameter("CleanCycleHeight", 100)
        self.cam.setParameter("CleanUntilTrigger", True)

        # Reaction to trigger
        self.cam.setParameter("TriggerDetermination", PicamTriggerDetermination["RisingEdge"])
        self.cam.setParameter("TriggerResponse", PicamTriggerResponse["ReadoutPerTrigger"])

        # Apply settings
        self.cam.sendConfiguration()

        ######################
        # Output data params #
        ######################
        self.shape = (512,512)
        self.dtype = np.int16

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
        data = self.cam.readNFrames(N = 1, timeout = 100000)[0].reshape(self.shape)

        # Get timestamp
        timestamp = time.time()-self.time_offset

        # Return the data and timestamp
        return [data, timestamp]

    def GetWarnings(self):
        return None