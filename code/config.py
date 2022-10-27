import os
import numpy as np


class Config_MAE_fMRI:
    # configs for fmri_pretrain.py
    def __init__(self):
    # --------------------------------------------
    # MAE for fMRI
        # Training Parameters
        self.lr = 2.5e-4
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 300
        self.warmup_epochs = 40
        self.batch_size = 50
        self.clip_grad = 0.8
        
        # Model Parameters
        self.mask_ratio = 0.35
        self.patch_size = 16
        self.embed_dim = 1024 # has to be a multiple of num_heads
        self.decoder_embed_dim = 512
        self.depth = 24
        self.decoder_depth = 8
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6
        # Project setting
        self.root_path = './mind-vis'
        self.output_path = './mind-vis/results'
        self.seed = 2022
        self.roi = 'VC'
        self.aug_times = 1
        self.accum_iter = 1 # Accumulate gradient iterations (for increasing the effective batch size under memory constraints)
        self.num_sub_limit = None
        self.group_name = 'patch_size'

        self.include_hcp = True
        self.include_kam = True
        self.include_shen = False
        
        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        # optua setting
        self.trial_numer = 0
        self.trial = None

        # distributed training
        self.world_size = 1
        self.local_rank = 0

class Config_MAE_finetune:
    def __init__(self):
        self.root_path = './mind-vis'
        self.pretrain_mae_path = os.path.join(self.root_path, 'pretrains/fmri_pretrain/checkpoints/checkpoint.pth') 
        self.kam_path = os.path.join(self.root_path, 'data/Kamitani/npz')
        self.bold5000_path = os.path.join(self.root_path, 'data/BOLD5000')
        self.dataset = 'Kamitani_2017' # Kamitani_2017 or Shen_2019 or BOLD5000
        self.include_nonavg_test = True
        self.kam_subs = ['sbj_3']
        self.bold5000_subs = ['CSI4']

        self.lr = 5.3e-5
        self.min_lr = 0.
        self.batch_size = 16 if self.dataset == 'Kamitani_2017' else 4 
        self.num_epoch = 15
        self.mask_ratio = 0.75 
        self.warmup_epochs = 2

        self.local_rank = 0
        self.output_path = '..'
        self.weight_decay = 0.05
        self.accum_iter = 1
        self.clip_grad = 0.8
        self.world_size = 1


class Config_Generative_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = './mind-vis'
        self.kam_path = os.path.join(self.root_path, 'data/Kamitani/npz')
        self.shen_path = os.path.join(self.root_path, 'data/Shen_2019/npz')
        self.bold5000_path = os.path.join(self.root_path, 'data/BOLD5000')
        self.roi = 'VC'
        self.patch_size = 16

        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/fmri_pretrain/22-08-2022-12:18:41/checkpoints/checkpoint-499.pth') # new 0.75
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/06-09-2022-15:52:47/checkpoints/checkpoint.pth') # new 0.75 fine-tuned
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/08-09-2022-01:14:46/checkpoints/checkpoint.pth') # new 0.75 fine-tuned 2 sub3
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/24-09-2022-19:42:32/checkpoints/checkpoint.pth') # new 0.75 fine-tuned 20 sub3

        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/17-09-2022-14:13:29/checkpoints/checkpoint.pth') # new 0.75 fine-tuned 2 sub1
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/17-09-2022-14:21:35/checkpoints/checkpoint.pth') # new 0.75 fine-tuned 2 sub2
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/17-09-2022-14:38:15/checkpoints/checkpoint.pth') # new 0.75 fine-tuned 2 sub4
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/17-09-2022-14:47:01/checkpoints/checkpoint.pth') # new 0.75 fine-tuned 2 sub5

        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/28-09-2022-09:55:42/checkpoints/checkpoint.pth') # with image loss

        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/17-09-2022-14:55:10/checkpoints/checkpoint.pth') # 0.75 large sub3
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/fmri_pretrain/15-09-2022-10:23:05/checkpoints/checkpoint.pth') # new large 0.75 
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/18-09-2022-10:02:59/checkpoints/checkpoint.pth') # new large 0.75  2
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/20-09-2022-21:31:48/checkpoints/checkpoint.pth') # new large 0.75  3

        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/23-09-2022-09:51:49/checkpoints/checkpoint.pth') # wrap 
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/23-09-2022-20:39:03/checkpoints/checkpoint.pth') # wrap 15
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/23-09-2022-21:02:19/checkpoints/checkpoint.pth') # wrap 20
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/23-09-2022-10:01:25/checkpoints/checkpoint.pth') # cut 
      
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/fmri_pretrain/28-08-2022-13:29:59/checkpoints/checkpoint.pth') # large 0.85
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/15-09-2022-10:35:59/checkpoints/checkpoint.pth') # large 0.85 fine-tuned

        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/27-09-2022-22:01:25/checkpoints/checkpoint.pth') # bold5000 CSI1
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/27-09-2022-22:44:48/checkpoints/checkpoint.pth') # bold5000 CSI2
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/27-09-2022-23:21:51/checkpoints/checkpoint.pth') # bold5000 CSI3
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/27-09-2022-23:36:14/checkpoints/checkpoint.pth') # bold5000 CSI4


        # abalation study
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/04-10-2022-00:00:33/checkpoints/checkpoint.pth') # emb 128
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/04-10-2022-00:08:58/checkpoints/checkpoint.pth') # emb 768
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/04-10-2022-00:17:26/checkpoints/checkpoint.pth') # emb 1024
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/04-10-2022-00:27:03/checkpoints/checkpoint.pth') # emb 1280

        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/05-10-2022-10:12:31/checkpoints/checkpoint.pth') # patch 32
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/05-10-2022-10:24:02/checkpoints/checkpoint.pth') # patch 64
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/06-10-2022-16:29:05/checkpoints/checkpoint.pth') # mask ratio 0.55
        
        # embed_dim
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/16-10-2022-22:28:07/checkpoints/checkpoint.pth') # patch size 32
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/16-10-2022-22:34:45/checkpoints/checkpoint.pth') # patch size 64
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/16-10-2022-22:41:55/checkpoints/checkpoint.pth') # patch size 256
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/16-10-2022-22:48:54/checkpoints/checkpoint.pth') # patch size 512

        # mask ratio
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/16-10-2022-22:58:37/checkpoints/checkpoint.pth') # mask ratio0.65
        # self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/16-10-2022-23:09:17/checkpoints/checkpoint.pth') # mask ratio0.45
        self.pretrain_mae_path = os.path.join(self.root_path, 'results/mae_finetune/16-10-2022-23:41:45/checkpoints/checkpoint.pth') # mask ratio0.35
        


        ######################################################################################
        # self.pretrain_gm_path = os.path.join(self.root_path, 'src/ldm/pretrains/semantic')
        self.pretrain_gm_path = os.path.join(self.root_path, 'src/ldm/pretrains/label2img')
        # self.pretrain_gm_path = os.path.join(self.root_path, 'src/ldm/pretrains/text2img-large')
        # self.pretrain_gm_path = os.path.join(self.root_path, 'src/ldm/pretrains/layout2img')
        
        self.dataset = 'Kamitani_2017' # Kamitani_2017 or Shen_2019 or BOLD5000
        self.kam_subs = ['sbj_3']
        self.shen_subs = ['sub-01']
        self.bold5000_subs = ['CSI1']

        self.group_name = 'ablation'
        self.img_size = 500 if self.dataset == 'Kamitani_2017' else 256

        np.random.seed(self.seed)
        self.test_category = None
        # self.test_category = list(np.random.choice(150, 50, replace=False))

        # finetune parameters
        self.batch_size1 = 5 if self.dataset == 'Kamitani_2017' else 25
        self.lr1 = 5.3e-5
        self.num_epoch_1 = 500
        
        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.mask_ratio = 0.0
        self.global_pool = False
        self.use_time_cond = True
        self.eval_avg = True

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None

        # distributed training
        self.world_size = 1
        self.local_rank = 0

        # resume check util
        self.model_meta = None
        self.checkpoint_path = None # os.path.join(self.root_path, 'results/generation/25-08-2022-08:02:55/checkpoint.pth')
        