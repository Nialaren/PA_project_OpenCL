import json


def load_from_file(path):
    fd = open(path)
    data = json.load(fd)
    fd.close()
    return data


def print_device_info(device):
    print 'Name: {0}'.format(device.name)
    print 'Vendor: {0}'.format(device.vendor)
    print 'Global memory size: {0}'.format(device.global_mem_size)
    print 'Max work group size: {0}'.format(device.max_work_group_size)
    print 'Size of device address space: {0}'.format(device.address_bits)
    print 'Extensions: {0}'.format(device.extensions)
