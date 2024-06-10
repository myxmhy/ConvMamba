from easydict import EasyDict as edict

def default_parser():
    default_values = {
    "setup_config":{
        "seed": 2023,
        "diff_seed": False,
        "per_device_train_batch_size": 24,
        "per_device_valid_batch_size": 24,
        "num_workers": 1,
        "method": "Trans",
        "max_epoch": 200,
        "lossfun": "CEL",
        "load_from": False,
        "if_continue": True,
        "regularization": 0.0,
        "if_display_method_info": True,
        "mem_log": True,
        "empty_cache": True,
        "metrics":["MSE", "RMSE", "MAE", "MRE", "SSIM", "MAX_RE",
                   "ACC", "PREC", "RECALL", "F1"],
        "drop_last": False,
    },
    "data_config": {
        "data_path": "./dataset.npy",
        "num_classes": 64,
        "samples_per_class": 600,
        "total_channels": 10,
        "select_channel": 10,
        "sequence_length": 3000,
        "train_rate": 0.6,
        "val_rate": 0.15
    },
    "optim_config": {
        "optim": "Adamw",
        "lr": 0.001,
        "filter_bias_and_bn": False,
        "log_step": 1,
        "opt_eps": "",
        "opt_betas": "",
        "momentum": 0.9,
        "weight_decay": 0.01,
        "early_stop_epoch": -1
    },
    "sched_config": {
        "sched": "onecycle",
        "min_lr": 1e-6,
        "warmup_lr": 1e-5,
        "warmup_epoch": 0,
        "decay_rate": 0.1,
        "decay_epoch": 100,
        "lr_k_decay": 1.0,
        "final_div_factor": 1e4
    },
    "model_config": {
        "nhead": 4,
        "num_encoder_layers": 3,
        "num_conv_layer": 6,
        "num_conv_filters":16
    },
    "ds_config":{
        "offload": False,
        "zero_stage": 0,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0
    }
    }

    return edict(default_values)
