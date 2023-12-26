import yaml, collections, copy
from pathlib import Path
    

def parse_exp_settings(experiment_path, only_this_sub_exp=''):
    with open(experiment_path, 'r') as file:
        experiment_settings = yaml.full_load(file)

    template_exp_settings = experiment_settings['template']
    template_exp_settings['experiment_name'] = experiment_settings['experiment_name']
    
    if only_this_sub_exp == template_exp_settings['sub_exp_name']:
        return [template_exp_settings]

    def deep_update(source, overrides):
        for key, value in overrides.items():
            if isinstance(value, collections.abc.Mapping) and value:
                returned = deep_update(source.get(key, {}), value)
                source[key] = returned
            else:
                source[key] = overrides[key]
        return source
    
    exp_list = [template_exp_settings]
    for sub_exp_overrides in experiment_settings.get('sub_experiments', []):
        sub_exp_settings = copy.deepcopy(template_exp_settings)
        sub_exp_settings = deep_update(sub_exp_settings, sub_exp_overrides)
        
        if only_this_sub_exp == sub_exp_settings['sub_exp_name']:
            return [sub_exp_settings]
        else:
            exp_list.append(sub_exp_settings)

    if only_this_sub_exp:
        return []
    return exp_list
   
    
def get_model_save_path(experiment_name, sub_exp_name):
    path_name = f'saved_models/{experiment_name}/{sub_exp_name}'
    Path(path_name).mkdir(parents=True, exist_ok=True)
    return path_name
