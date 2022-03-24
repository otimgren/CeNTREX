"""
Driver for the Zaber T-NA08A50 linear actuator.
"""
import logging
import time

from zaber_motion import Library, LogOutputMode, Tools, Units
from zaber_motion.binary import Connection


class ZaberLinAct:
    def __init__(
        self, time_offset, COM_port: str, zero_position: float = 25,
    ):
        self.time_offset = time_offset
        self.COM_port = COM_port
        self.zero_position = zero_position
        self.unit = Units.LENGTH_MILLIMETRES

        self.dtype = ("float", "float")
        self.shape = (2,)

        # Connect to device
        try:
            self.conn = Connection.open_serial_port(self.COM_port)
            devices = self.conn.detect_devices()
            self.device = devices[0]
        except Exception as e:
            logging.warning("ZaberLinAct error in initial connection : " + str(e))
            self.verification_string = "False"
            self.__exit__()

        # Set device to its zero position
        self.GoToZeroGUI()

        # Find device serial number to use as verification string
        self.verification_string = str(self.device.serial_number)

        self.warnings = []
        self.new_attributes = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.conn.close()
        return

    #######################################################
    ################ CeNTREX DAQ Commands #################
    #######################################################

    def CreateWarning(self, warning):
        warning_dict = {"message": warning}
        self.warnings.append([time.time(), warning_dict])

    def GetWarnings(self):
        warnings = self.warnings.copy()
        self.warnings = []
        return warnings

    def ReadValue(self):
        val = [
            time.time() - self.time_offset,
            self.device.get_position(self.unit) - self.zero_position,
        ]
        return val

    #######################################################
    ##################### GUI Commands ####################
    #######################################################

    def GetPositionGUI(self) -> float:
        """
        Returns the current position of the actuator.
        """
        return f"{self.device.get_position(self.unit) - self.zero_position:.3f}"

    def SetPositionGUI(self, position: float) -> float:
        """
        Sets the actuator to an absolute position.
        """
        return f"{self.SetPosition(position):.3f}"

    def GoToZeroGUI(self) -> float:
        """
        Sets the position to the zero position
        """
        return self.SetPosition(0)

    #######################################################
    ################## Device Commands ####################
    #######################################################
    def SetPosition(self, position: float) -> float:
        """
        Sets the actuator to an absolute position and returns the position.
        """
        return self.device.move_absolute(position + self.zero_position, self.unit)
