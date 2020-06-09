import torch
import os

__all__ = ['cifar_10_loader']

def cifar_10_loader(data_root='./data/cifar-10-batches-py', train=True):
    train_file_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_file_list = ['test_batch']
    file_list = train_file_list if train else test_file_list

    img_list = []
    ann_list = []
    for file in file_list:
        data = unpickle(os.path.join(data_root, file))
        img = data[b'data']
        ann = data[b'labels']

        img_list.append(torch.tensor(img).float())
        ann_list.append(torch.tensor(ann).float())

    data_dict = {
        'img' : torch.cat(img_list, dim=0),
        'ann' : torch.cat(ann_list, dim=0)
    }

    return data_dict

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
