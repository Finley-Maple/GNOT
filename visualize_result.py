#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import pickle
import torch
import numpy as np
import torch.nn as nn
import dgl
import matplotlib.pyplot as plt
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import get_seed, get_num_params
from args import get_args
from data_utils import get_dataset, get_model, get_loss_func, MIODataLoader
from train import validate_epoch
from utils import plot_heatmap
import os

if __name__ == "__main__":

    save_dir = './data/checkpoints'
    model_name = 'heat2d_all_CGPTrel2_1029_17_33_50.pt'
    model_path = save_dir + '/' + model_name
    result = torch.load(model_path,map_location='cpu')


    args = result['args']

    model_dict = result['model']

    vis_component = 0 if args.component == 'all' else int(args.component)

    device = torch.device('cpu')


    kwargs = {'pin_memory': False} if args.gpu else {}
    get_seed(args.seed, printout=False)

    train_dataset, test_dataset = get_dataset(args)

    test_sampler = SubsetRandomSampler(torch.arange(len(test_dataset)))

    test_loader = MIODataLoader(test_dataset, sampler=test_sampler, batch_size=1, drop_last=False)

    loss_func = get_loss_func(args.loss_name, args, regularizer=True,  normalizer=args.normalizer)
    metric_func = get_loss_func(args.loss_name, args , regularizer=False, normalizer=args.normalizer)

    model = get_model(args,)


    model.load_state_dict(model_dict)

    model.eval()
    with torch.no_grad():
        #### test single case
        idx = 0
        g, u_p, g_u =  list(iter(test_loader))[idx]
        # u_p = u_p.unsqueeze(0)      ### test if necessary
        out = model(g, u_p, g_u)

        x, y = g.ndata['x'][:,0].cpu().numpy(), g.ndata['x'][:,1].cpu().numpy()
        pred = out[:,vis_component].squeeze().cpu().numpy()
        target =g.ndata['y'][:,vis_component].squeeze().cpu().numpy()
        err = pred - target
        print(pred)
        print(target)
        print(err)
        print(np.linalg.norm(err)/np.linalg.norm(target))

        fig_save_dir = './data/figures'
        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)
            
        fig_save_path = fig_save_dir + '/' + model_name.replace('.pt','.png')
        #### choose one to visualize
        cm = plt.cm.get_cmap('rainbow')
        
        plot_heatmap(x, y, pred,cmap=cm,path=fig_save_path.replace('.png','_pred_heatmap.png'))
        plot_heatmap(x, y, target,cmap=cm,path=fig_save_path.replace('.png','_target_heatmap.png'))


        plt.figure()
        plt.scatter(x, y, c=pred, cmap=cm, s=2)
        plt.colorbar()
        # plt.savefig()
        plt.show()
        
        
        plt.figure()
        plt.scatter(x, y, c=err, cmap=cm,s=2)
        plt.colorbar()
        plt.savefig(fig_save_path.replace('.png','_err.png'))
        # plt.show()
        
        plt.scatter(x, y, c=target, cmap=cm, s=2)
        plt.colorbar()
        # plt.savefig()
        plt.show()





