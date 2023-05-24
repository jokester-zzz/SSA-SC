import hashlib
import os
import torch


def get_md5(filename):
    '''

    '''
    hash_obj = hashlib.md5()
    with open(filename, 'rb') as f:
        hash_obj.update(f.read())
    return hash_obj.hexdigest()

def dict_to(_dict, device):
    # for key, value in _dict.items():
    #   if type(_dict[key]) is dict:
    #     _dict[key] = dict_to(_dict[key], device)
    #   else:
    #     _dict[key] = _dict[key].to(device=device)

    _dict['3D_OCCUPANCY'] = _dict['3D_OCCUPANCY'].to(device=device)
    _dict['3D_OCCLUDED'] = _dict['3D_OCCLUDED'].to(device=device)
    _dict['3D_LABEL']['1_1'] = _dict['3D_LABEL']['1_1'].to(device=device)
    _dict['feature'] = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in _dict['feature']]
    _dict['grid_ind'] = [torch.from_numpy(i[:, 0:2]).type(torch.FloatTensor).to(device) for i in _dict['voxel_ind']]
    _dict['voxel_ind'] = [torch.from_numpy(i[:, :]).type(torch.LongTensor).to(device) for i in _dict['voxel_ind']]
    _dict['processed_label'] = _dict['processed_label'].type(torch.LongTensor).to(device=device)
    # print(torch.unique(_dict['processed_label']))

    return _dict


def _remove_recursively(folder_path):
    '''
    Remove directory recursively
    '''
    if os.path.isdir(folder_path):
        filelist = [f for f in os.listdir(folder_path)]
        for f in filelist:
            os.remove(os.path.join(folder_path, f))
    return


def _create_directory(directory):
    '''
    Create directory if doesn't exists
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
    return
