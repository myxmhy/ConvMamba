import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import print_log
import os
import os.path as osp
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
config = {
    "font.family":'Times New Roman',
    "font.size": 13,
    "mathtext.fontset": 'stix',
    "mathtext.rm": 'Times New Roman',
    "mathtext.it": 'Times New Roman:italic',
    "mathtext.bf": 'Times New Roman:bold'
}
rcParams.update(config)

def plot_learning_curve(loss_record, model_path, dpi=300, title='', dir_name = "pic"):
    ''' Plot learning curve of your DNN (train & valid loss) '''
    total_steps = len(loss_record['train_loss'])
    x_1 = range(total_steps)
    plt.semilogy(x_1, loss_record['train_loss'], c='tab:red', label='train')
    if len(loss_record['valid_loss']) != 0:
        x_2 = x_1[::len(loss_record['train_loss']) // len(loss_record['valid_loss'])]
        plt.semilogy(x_2, loss_record['valid_loss'], c='tab:cyan', label='valid')
    plt.xlabel('Training steps', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.title('Learning curve of {}'.format(title), fontsize=15)
    plt.legend()

    pic_name = f'loss_record.png'
    pic_folder = osp.join(model_path, dir_name)
    os.makedirs(pic_folder, exist_ok=True)
    pic_path =osp.join(pic_folder, pic_name)
    print_log(f'Simulation picture saved in {pic_path}')
    plt.savefig(pic_path, dpi=dpi, bbox_inches='tight')
    plt.close()
