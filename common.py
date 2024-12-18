#!/usr/bin/env python
# coding: utf-8

# Model & training arguments are defined in this file


class Config():
    
    save_ckpt_dir = '/home/jayesh/Documents/CNN/train_model'
    result_out_dir = '/home/jayesh/Documents/CNN/results'
    data_dir = '/home/jayesh/Documents/CNN/data'
    ckpt_dir = None
    use_gpu = True
    use_dgconv = True
    fix_groups = 1     
    
    num_replicates = 5
    seed = 42
    dataset = 'berlin'
    
    if dataset == 'houston':
        # hyperparams for unet
        model = 'unet'
        mask_undefined = True
        num_classes = 20
        epochs = 300
        lr = 0.001
        lr_schedule = None
        optimizer = 'adam'
        batch_size = 12
        use_init = False
        sample_h = sample_w = 128
        # hyperparams for fusion_fcn
#         model = 'fusion_fcn'
#         mask_undefined = True
#         num_classes = 20
#         epochs = 4000
#         lr = 0.001
#         lr_schedule = None
#         optimizer = 'adam'
#         batch_size = 2
#         use_init = True
#         sample_h = 1202
#         sample_w = 300
    elif dataset == 'berlin':
        mask_undefined = False
        num_classes = 8
        # hyperparameters for resnet18
        model = 'resnet18'
        epochs = 300
        lr = 0.001
        lr_schedule = None
        optimizer = 'sgd'
        batch_size = 64
        use_init = False
        sample_radius = 8
        # hyperparameters for resnet50
#         model = 'resnet50'
#         epochs = 400
#         lr = 0.001
#         lr_schedule = [300]
#         optimizer = 'adam'
#         batch_size = 64
#         use_init = False
#         sample_radius = 8
    elif dataset == 'muufl':
        mask_undefined = False
        num_classes = 11
        use_init = True
        # hyperparams for resnet18
        model = 'resnet18'
        sample_radius = 5
        epochs = 300
        lr = 0.02
        lr_schedule = [200, 240]
        optimizer = 'sgd'
        batch_size = 48
        # hyperparams for resnet50
#         model = 'resnet50'
#         sample_radius = 8
#         epochs = 400
#         lr = 0.01
#         lr_schedule = [300, 350]
#         optimizer = 'adam'
#         batch_size = 64
        # hyperparams for tb_cnn
#         model = 'tb_cnn'
#         sample_radius = 5
#         epochs = 300
#         lr = 0.001
#         lr_schedule = None
#         optimizer = 'adam'
#         batch_size = 48
    

