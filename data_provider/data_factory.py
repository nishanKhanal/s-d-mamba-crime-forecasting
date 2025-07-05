from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred, Dataset_Crime, Dataset_Synthetic, Dataset_Synthetic_Through_Rotation, Dataset_Synthetic_2D
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'crime': Dataset_Crime,
    'synthetic': Dataset_Synthetic,
    'synthetic_rotate': Dataset_Synthetic_Through_Rotation,
    'synthetic_2d': Dataset_Synthetic_2D,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    kwargs = {
        "root_path" : args.root_path,
        "data_path" : args.data_path,
        "flag" : flag,
        "size" : [args.seq_len, args.label_len, args.pred_len],
        "features" : args.features,
        "target" : args.target,
        "timeenc" : timeenc,
        "freq" : freq,
    }

    if 'synthetic' in args.data:
        kwargs['syn_data_params'] = {
            'N': args.height * args.width if args.data == 'synthetic_2d' else args.num_nodes,
            'T': args.num_time_steps,
            'mode': args.mode,
            'w': args.w,
            'dt': args.dt,
            'noise_strength': args.noise_strength,
            'theta_deg': args.theta_deg,
            'pad_mode': args.pad_mode,
            
            # for synthetic_2d
            'H': args.height,
            'W': args.width,
            'alpha': args.alpha,
            'noise_std': args.noise_std
        }
    data_set = Data(**kwargs)
    
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
