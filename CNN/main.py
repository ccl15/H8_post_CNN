import argparse, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from modules.experiment_helper import parse_exp_settings, get_model_save_path
from modules.training_helper import set_up_tensorflow, get_summary_writer, \
    get_TFRecord_dataset, create_model_by_exp_settings
from modules.model_trainer import train_model


def main(exp_path, GPU_limit, omit_completed):
    # environment setting
    set_up_tensorflow(GPU_limit)

    # parse yaml to get experiment settings 
    exp_list = parse_exp_settings(exp_path)
    
    for sub_exp_settings in exp_list:
        exp_name = sub_exp_settings['experiment_name']
        sub_exp_name = sub_exp_settings['sub_exp_name']
        log_path = f'logs/{exp_name}/{sub_exp_name}'
        
        print(f'Executing sub-experiment: {sub_exp_name} ...')
        if omit_completed and os.path.isdir(log_path):
            print('Sub-experiment already done, skipped ~~~~')
            continue
        
        # reset tensorflow 

        # log and model saved path setting
        summary_writer = get_summary_writer(log_path)
        model_save_path = get_model_save_path(exp_name, sub_exp_name)
        
        # load data and create model. ^train_helper
        load_path = sub_exp_settings.get('load_path', '')
        model = create_model_by_exp_settings(sub_exp_settings['model'], sub_exp_settings['model_setting'], load_path)
        if 'datasets' not in locals():
            datasets = get_TFRecord_dataset(**sub_exp_settings['data'])
       
        # training. ^model_trainer
        train_model(
            model,
            datasets,
            summary_writer,
            model_save_path,
            **sub_exp_settings['train_setting'],
        )
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_path', help='name of the experiment setting')
    parser.add_argument('--GPU_limit', type=int, default=20000)
    parser.add_argument('-G', '--CUDA_VISIBLE_DEVICES', type=str, default='')
    parser.add_argument('--omit', action='store_true')
    args = parser.parse_args()
    
    if args.CUDA_VISIBLE_DEVICES:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    
    main(args.exp_path, args.GPU_limit, args.omit)
