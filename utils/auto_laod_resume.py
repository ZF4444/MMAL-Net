import os
import torch
from config import init_lr
from collections import OrderedDict

def auto_load_resume(model, path, status):
    if status == 'train':
        pth_files = os.listdir(path)
        nums_epoch = [int(name.replace('epoch', '').replace('.pth', '')) for name in pth_files if '.pth' in name]
        if len(nums_epoch) == 0:
            return 0, init_lr
        else:
            max_epoch = max(nums_epoch)
            pth_path = os.path.join(path, 'epoch' + str(max_epoch) + '.pth')
            print('Load model from', pth_path)
            checkpoint = torch.load(pth_path)
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            epoch = checkpoint['epoch']
            lr = checkpoint['learning_rate']
            print('Resume from %s' % pth_path)
            return epoch, lr
    elif status == 'test':
        print('Load model from', path)
        checkpoint = torch.load(path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if 'module.' == k[:7]:
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        epoch = checkpoint['epoch']
        print('Resume from %s' % path)
        return epoch