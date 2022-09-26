"""
Naming conventions, some specs, hyperparameters and stuff
"""

def resolve_dataset(dataset_name):
    # That can't be defined here
    experiment_config = {
        'dataset_name': dataset_name,
    }
    if dataset_name=='CIFAR10':
        experiment_config['code_dim'] = 10
        experiment_config['model_teacher'] = "resnet34"
        experiment_config['model_student'] = "resnet18"
        experiment_config['inputs'] = 3
        experiment_config['channels'] = 3
        experiment_config['batch_size'] = 256
    return experiment_config


def resolve_teacher_name(experiment_config):
    model_name = "teacher_"
    model_name += "{}_{}".format(
        experiment_config['dataset_name'],
        experiment_config['model_teacher']
    )
    model_name += ".model"
    return model_name
