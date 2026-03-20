import torch
import argparse
import datetime
import os
import sys
from easydict import EasyDict as edict
from utils.misc import make_folder

def parse_configs():
    parser = argparse.ArgumentParser(description='PIDA')
    parser.add_argument('--seed', type=int, default=2024,
                    help='re-produce the results with seed random')
    parser.add_argument('--working-dir', type=str, default='../', metavar='PATH',
                        help='the ROOT working directory')
    parser.add_argument('--saved_fn', type=str, default='logs', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('--no_val', action='store_true',
                    help='If true, use all data for training, no validation set')
    parser.add_argument('--no_test', action='store_true',
                        help='If true, dont evaluate the model on the test set')
    parser.add_argument('--test', action='store_true',
                        help='If true, dont oversampling the event')
    parser.add_argument('--val-size', type=float, default=0.2,
                    help='The size of validation set')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='mini-batch size (default: 8), this is the total'
                            'batch size of all GPUs on the current node when using'
                            'Data Parallel or Distributed Data Parallel')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of threads for loading data')
    parser.add_argument('--distributed', action='store_true', 
                       help="If using multiple GPUs for distributed training")
    parser.add_argument('--print_freq', type=int, default=100, metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('--checkpoint_freq', type=int, default=1, metavar='N',
                        help='frequency of saving checkpoints (default: 1)')
    parser.add_argument('--earlystop_patience', type=int, default=None, metavar='N',
                        help='Early stopping the training process if performance is not improved within this value')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the image of testing phase will be saved')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    ### build backbone network
    parser.add_argument('--backbone_choice', type=str, default="single",
                        help="single means single feature map, multi for multi level feature map")
    parser.add_argument('--backbone_pretrained', type=bool, default=True,
                        help="whether backbone should be pretrained")
    parser.add_argument('--backbone_out_channels', type=int, default=2048,
                        help="which level the backbone to choose to output feature map")
    ### build transformer
    parser.add_argument('--transfromer_dmodel', type=int, default=512,
                        help="dimension of the transformer model")
    parser.add_argument('--transformer_nhead', type=int, default=8,
                        help="number of multihead in transformer")
    parser.add_argument('--num_feature_levels', type=int, default=1,
                        help="number of feature map from backbone network")
    ### build detector
    parser.add_argument('--num_classes', type=int, default=1,
                        help="number of classes expected in detector")
    parser.add_argument('--num_queries', type=int, default=20,
                        help="numebr of queries in the transformer")
    parser.add_argument('--model_choice', type=str, required=True,
                        help="choice of model including wasb, tracknet, tracknetv2, mamba")
    parser.add_argument(
        '--weighting_list', 
        type=int, 
        nargs=4,  # This makes sure 4
        metavar=('vis_0', 'vis_1', 'vis_2', 'vis_3'),  # Labels for help message
        default=[1, 2, 2, 3],  # Default as a list
        help="Specify the weighting of loss bce(e.g., --weighting_list 1 2 2 3)"
    )
    parser.add_argument('--num_channels', type=int, default=64, 
                        help="number of channels for model")
    
    
    ####################################################################
    ##############     Demonstration configurations     ###################
    ####################################################################
    parser.add_argument('--video_path', type=str, default=None, metavar='PATH',
                        help='the path of the video that needs to demo')
    parser.add_argument('--output_format', type=str, default='text', metavar='PATH',
                        help='the type of the demo output')
    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_demo_output', action='store_true',
                        help='If true, the image of demonstration phase will be saved')
    

    ####################################################################
    ##############     Training strategy            ###################
    ####################################################################
    parser.add_argument('--num_frames', type=int, default=9, metavar='N',
                        help='number of frames into model')
    parser.add_argument('--interval', type=int, default=1, metavar='N',
                        help='number of intervals between frames')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='the starting epoch')
    parser.add_argument('--num_epochs', type=int, default=30, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--minimum_lr', type=float, default=1e-7, metavar='MIN_LR',
                        help='minimum learning rate during training')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, metavar='WD',
                        help='weight decay (default: 0)')
    parser.add_argument('--optimizer_type', type=str, default='adam', metavar='OPTIMIZER',
                        help='the type of optimizer, it can be sgd or adam or adamw')
    parser.add_argument('--loss_function', type=str, default='WBCE',
                        help='the type of loss function, it can be BCE or WBCE')
    parser.add_argument('--lr_type', type=str, default='plateau', metavar='SCHEDULER',
                        help='the type of the learning rate scheduler (steplr or ReduceonPlateau)')
    parser.add_argument('--lr_factor', type=float, default=0.5, metavar='FACTOR',
                        help='reduce the learning rate with this factor')
    parser.add_argument('--lr_step_size', type=int, default=5, metavar='STEP_SIZE',
                        help='step_size of the learning rate when using steplr scheduler')
    parser.add_argument('--lr_patience', type=int, default=3, metavar='N',
                        help='patience of the learning rate when using ReduceoPlateau scheduler')
    parser.add_argument('--occluded_prob', type=float, default=0.5, metavar='N',
                        help='occluded probability of ball')
    parser.add_argument('--ball_size', type=int, default=5, metavar='N',
                        help='ball size to determine percision recall and f1 socre')
    parser.add_argument('--dataset_choice', type=str, default='tt', metavar='DC',
                        help='which dataset to use tt for table tennis, tennis for tennis')
    parser.add_argument('--event', action='store_true',
                        help='If true, use event dataset, which is only available in tt dataset! ')
    parser.add_argument('--mimo', action='store_true',
                        help='If true, use mimo setting, which means multiple in and multiple out ')
    parser.add_argument('--bidirect', action='store_true',
                        help='If true, ball frame will be middle not last, if false will be last')
    parser.add_argument('--sequential', action='store_true',
                        help='If true, sequential dataset will be used, the data will contains all ball locations for all frames')
    parser.add_argument('--smooth_labelling', action='store_true',
                        help='If true, smooth labelling is true')
    parser.add_argument(
        '--img_size', 
        type=int, 
        nargs=2,  # This makes sure two integers are provided (for width and height)
        metavar=('height', 'wdith'),
        default=(288, 512),
        help="Specify the new image size as width and height (e.g., --img_size 540 960)"
    )
    parser.add_argument('--resize', type=tuple,
                        help='resize the image to this size, it should be a tuple of (width,height), e.g., (398, 224)')
    parser.add_argument('--new_data', action='store_true',
                        help='If true, use new data for training, which is the data from TTA dataset')
    

    ####################################################################
    ##############     Distributed Data Parallel            ############
    ####################################################################
    parser.add_argument('--world_size', default=-1, type=int, metavar='N',
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, metavar='N',
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:29500', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    configs = edict(vars(parser.parse_args()))
    ####################################################################
    ############## Hardware configurations ############################
    ####################################################################
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda')
    
    configs.ngpus_per_node = torch.cuda.device_count()

    configs.pin_memory = True

    configs.org_size = (1080, 1920)
    configs.fps = 25

    configs.results_dir = os.path.join(configs.working_dir, 'results')
    configs.logs_dir = os.path.join(configs.working_dir, 'logs', configs.saved_fn)
    configs.checkpoints_dir = os.path.join(configs.working_dir, 'checkpoints', configs.saved_fn)
    configs.frame_dir = os.path.join('/home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/demo')

    ####################################################################
    ##############     Data configs            ###################
    ####################################################################

    # configs.dataset_dir = os.path.join(configs.working_dir, 'dataset')

    configs.dataset_dir = os.path.join('/home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/data/', 'tt')
    configs.train_game_list = ['game_1', 'game_2', 'game_3', 'game_4', 'game_5']
    configs.test_game_list = ['test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6', 'test_7']
    configs.events_dict = {
        'bounce': 0,
        'net': 1,
        'empty_event': 2
    }
    configs.events_weights_loss_dict = {
        'bounce': 1.,
        'net': 3.,
    }

    ###################################################################
    ##############          Tennis dataset
    ####################
    configs.tennis_dataset_dir = os.path.join('/home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/data/', 'tennis_data')
    configs.tennis_train_game_list = ['game1', 'game2', 'game3', 'game4', 'game5', 'game6', 'game7', 'game8']
    configs.tennis_test_game_list = ['game9', 'game10']


    configs.badminton_dataset_dir = os.path.join('/home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/data/badminton/TrackNetV2')
    configs.badminton_train_game_list = ['Amateur', 'Professional']
    configs.badminton_test_game_list = ['Test']

    configs.tta_dataset_dir = os.path.join('/home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/data/tta_dataset')
    configs.tta_training_match_list = ['24Paralympics_FRA_F9_Lei_AUS_v_Xiong_CHN', '24Paralympics_FRA_M4_Addis_AUS_v_Chaiwut_THA', '24Paralympics_FRA_M9_Ma_AUS_v_Didier_FRA']
    configs.tta_test_match_list = ['24Paralympics_FRA_M4_Addis_AUS_v_Chaiwut_THA']

    configs.tta_tracking_dataset_dir = os.path.join('/home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/data/tta_tracking')

    make_folder(configs.checkpoints_dir)
    make_folder(configs.logs_dir)
    make_folder(configs.results_dir)

    if configs.save_test_output:
        configs.saved_dir = os.path.join(configs.results_dir, configs.saved_fn)
        make_folder(configs.saved_dir)

    if configs.save_demo_output:
        configs.save_demo_dir = os.path.join(configs.results_dir, 'demo', configs.saved_fn)
        make_folder(configs.save_demo_dir)


    return configs


if __name__ == "__main__":
    configs = parse_configs()
    print(configs)

    print(datetime.date.today())
    print(datetime.datetime.now().year)
