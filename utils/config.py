import yaml
import shutil
from easydict import EasyDict
import os


def log_args_to_file(args, pre='args', logger=None):
    """Log args to file only (not terminal). Uses DEBUG level to skip StreamHandler."""
    if logger is None:
        return
    for key, val in args.__dict__.items():
        logger.debug(f'{pre}.{key} : {val}')


def log_config_to_file(cfg, pre='config', logger=None):
    """Log config to file only (not terminal). Uses DEBUG level to skip StreamHandler."""
    if logger is None:
        return
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.debug(f'{pre}.{key} : {val}')


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(val, 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except Exception as e:
                        raise e
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
        else:
            config[key] = EasyDict()
            merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            new_config = yaml.load(f)
    merge_new_config(config, new_config)
    return config


def get_config(args, logger=None):
    config = cfg_from_yaml_file(args.config)

    # Override max_epoch from command line if specified
    if hasattr(args, 'max_epoch') and args.max_epoch is not None:
        original_epoch = config.get('max_epoch', None)
        config.max_epoch = args.max_epoch
        if logger and original_epoch is not None:
            logger.info(f'Overriding max_epoch: {original_epoch} -> {args.max_epoch} (from --max_epoch argument)')

    # Copy config file to experiment directory
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    shutil.copy(args.config, config_path)
    if logger:
        logger.info(f'Config copied to {config_path}')

    return config
