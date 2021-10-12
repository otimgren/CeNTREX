import h5py
from pathlib import Path
from typing import Tuple

def load_measurement_data(fname, dset_name):
    # loading data from HDF file
    with h5py.File(fname, 'r') as f:
        laserlock = f[dset_name]['readout']['Laser Lock'][()]
        wavelength = f[dset_name]['readout']['Wavelength'][()]
        mirror = f[dset_name]['readout']['ZaberTMM'][()]
        pxie_time = {}
        pxie = {}
        for dset in f[dset_name]['readout']['PXIe-5171']:
            if 'events' not in dset:
                pxie[int(dset.split('_')[-1])] = f[dset_name]['readout']['PXIe-5171'][dset][()]
                pxie_time[int(dset.split('_')[-1])] = f[dset_name]['readout']['PXIe-5171'][dset].attrs['ch0 : timestamp']

    return pxie, pxie_time, laserlock, wavelength, mirror

def load_measurement_data_devices(fname, dset_name, devices = ('Laser Lock',
                          'Wavelength', 'ZaberTMM'), ts_ch = 'ch0'):
    # loading data from HDF file
    with h5py.File(fname, 'r') as f:
        data_devices = {}
        for device in devices:
            data_devices[device] = f[dset_name]['readout'][device][()]
        pxie_time = {}
        pxie = {}
        for dset in f[dset_name]['readout']['PXIe-5171']:
            if 'events' not in dset:
                pxie[int(dset.split('_')[-1])] = f[dset_name]['readout']['PXIe-5171'][dset][()]
                pxie_time[int(dset.split('_')[-1])] = f[dset_name]['readout']['PXIe-5171'][dset].attrs[f'{ts_ch} : timestamp']

    return pxie, pxie_time, data_devices

def load_measurement_data_devices_attrs(fname, dset_name, devices = ('Laser Lock',
                          'Wavelength', 'ZaberTMM'), ts_ch = 'ch0', devices_path = 'readout',pxie_path = 'readout'):
    # loading data from HDF file
    with h5py.File(fname, 'r') as f:
        data_devices = {}
        for device in devices:
            data_devices[device] = f[dset_name][devices_path][device][()]
        pxie_time = {}
        pxie = {}
        pxie_attrs = {}
        for dset in f[dset_name][pxie_path]['PXIe-5171']:
            if 'events' not in dset:
                pxie[int(dset.split('_')[-1])] = f[dset_name][pxie_path]['PXIe-5171'][dset][()]
                pxie_time[int(dset.split('_')[-1])] = f[dset_name][pxie_path]['PXIe-5171'][dset].attrs[f'{ts_ch} : timestamp']
                pxie_attrs[int(dset.split('_')[-1])] = {k:v for k,v in f[dset_name][pxie_path]['PXIe-5171'][dset].attrs.items()}
    return pxie, pxie_time, pxie_attrs, data_devices

def load_camera_data(fpath: Path, dset_name: str, camera_path = 'camera_test') \
    -> Tuple[dict, dict]:
    """
    Loads camera data from hdf file.
    """
    camera_data = {}
    camera_time = {}
    with h5py.File(fpath, 'r') as f:
        for dset in f[dset_name][camera_path]['PIProEM512Excelon']:
            if 'events' not in dset:
                camera_data[int(dset.split('_')[-1])] = f[dset_name][camera_path]['PIProEM512Excelon'][dset][()]
                camera_time[int(dset.split('_')[-1])] = f[dset_name][camera_path]['PIProEM512Excelon'][dset].attrs[f'timestamp']

    return camera_data, camera_time
