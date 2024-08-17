'''
Author: zhangshd
Date: 2024-08-09 16:49:54
LastEditors: zhangshd
LastEditTime: 2024-08-17 19:17:24
'''


from sacred import Experiment

ex = Experiment("mof_stability", save_git_info=False)


@ex.config
def cfg():
    # Basic Training Control
    batch_size = 8  # Batch size
    num_workers = 2  # Number of worker processes for data loading
    random_seed = 42  # Random seed
    accelerator = "gpu"  # Accelerator type
    devices = 1  # Number of devices
    max_epochs = 1000  # Maximum number of training epochs
    limit_train_batches = None  # Limit on training batches
    limit_val_batches = None  # Limit on validation batches
    auto_lr_bs_find = False  # Auto learning rate and batch size finder flag
    progress_bar = True  # Progress bar display flag

    # Loss Function
    focal_alpha = 0.25  # Focal loss alpha parameter
    focal_gamma = 2  # Focal loss gamma parameter

    # Optimizer
    optim = 'adam'  # Optimizer type
    lr = 1e-3  # Learning rate
    weight_decay = 1e-5  # Weight decay
    momentum = 0.9  # Momentum parameter
    optim_config = "coarse"  # Optimizer configuration: coarse or fine
    group_lr = False  # Group learning rate flag
    lr_mult = 10  # Learning rate multiplier for multi-task learning heads
    

    # LR Scheduler
    lr_scheduler = 'reduce_on_plateau'  # Learning rate scheduler type: multi_step, cosine, reduce_on_plateau
    lr_decay_steps = 20  # Learning rate decay steps
    lr_milestones = [10, 20, 30, 50]  # Learning rate milestones
    lr_decay_rate = 0.8  # Learning rate decay rate
    lr_decay_min_lr = 1e-6  # Minimum learning rate for decay
    max_steps = -1  # Maximum number of training steps
    decay_power = (
        1  # Power of polynomial decay function
                   ) 
    warmup_steps = 2
    

    # Restart Control
    load_best = False  # Load best model flag
    load_dir = None  # Directory to load the model from
    load_ver = None  # Version of the model to load
    load_v_num = None  # Number of the model to load

    # Training Info
    log_dir = 'logs'  # Log directory
    patience = 50  # Patience
    min_delta = 0.001  # Minimum change
    monitor = 'val_Metric'  # Monitoring metric
    mode = 'max'  # Mode
    eval_freq = 10  # valset Evaluation frequency

    # Data Module Hyperparameters
    max_num_nbr = 10  # Maximum number of neighbors
    radius = 8  # Radius
    dmin = 0  # Minimum distance
    step = 0.2  # Step
    use_cell_params = False  # Use cell parameters flag
    use_extra_fea = False  # Use extra features flag
    task_weights = None
    augment = False  # Data augmentation flag
    max_sample_size = {
                "train": 2004,
                "val": 501,
            }

    # Model Hyperparameters
    model_name = 'fcnn'  # Model name
    atom_fea_len = 128  # Atom feature length
    extra_fea_len = 128  # Extra feature length
    h_fea_len = 256  # Hidden feature length
    n_conv = 3  # Number of convolutional layers
    n_h = 4  # Number of hidden layers
    att_S = 64  # S parameter
    dropout_prob = 0.0  # Dropout probability
    att_pooling = True # Attention pooling flag
    task_norm = True  # Task normalization flag
    dwa_temp = 2.0  # DWA temperature parameter
    dwa_alpha = 0.8  # DWA alpha parameter



@ex.named_config
def cgcnn():
    model_name = 'cgcnn'  # Model name
    atom_fea_len = 64  # Atom feature length
    extra_fea_len = 128  # Extra feature length
    h_fea_len = 128  # Hidden feature length
    n_conv = 3  # Number of convolutional layers
    n_h = 2  # Number of hidden layers
    dropout_prob = 0.0  # Dropout probability
    use_extra_fea = False  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    atom_layer_norm = True  # Atom layer normalization flag

@ex.named_config
def cgcnn_raw():
    model_name = 'cgcnn_raw'  # Model name
    atom_fea_len = 64  # Atom feature length
    extra_fea_len = 128  # Extra feature length
    h_fea_len = 128  # Hidden feature length
    n_conv = 3  # Number of convolutional layers
    n_h = 2  # Number of hidden layers
    dropout_prob = 0.0  # Dropout probability
    use_extra_fea = False  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    atom_layer_norm = True  # Atom layer normalization flag

@ex.named_config
def att_cgcnn():
    model_name = 'att_cgcnn'  # Model name
    atom_fea_len = 64  # Atom feature length
    extra_fea_len = 128  # Extra feature length
    h_fea_len = 128  # Hidden feature length
    n_conv = 3  # Number of convolutional layers
    n_h = 2  # Number of hidden layers
    dropout_prob = 0.0  # Dropout probability
    use_extra_fea = False  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    atom_layer_norm = True  # Atom layer normalization flag
    task_att_type = 'self'  # Attention type: self or external
    att_S = 64  # S parameter of external attention

@ex.named_config
def cgcnn_uni_atom():
    model_name = 'cgcnn_uni_atom'  # Model name
    atom_fea_len = 64  # Atom feature length
    extra_fea_len = 128  # Extra feature length
    max_graph_len = 300  # Maximum number of atoms in a graph
    h_fea_len = 128  # Hidden feature length
    n_conv = 3  # Number of convolutional layers
    n_h = 2  # Number of hidden layers
    dropout_prob = 0.0  # Dropout probability
    use_extra_fea = False  # Use extra features flag
    use_cell_params = False  # Use cell parameters flag
    atom_layer_norm = True  # Atom layer normalization flag
    task_att_type = 'self'  # Attention type: self or external
    att_S = 64  # S parameter of external attention
    reconstruct = False  # Reconstruct atom features into fixed length gragph representation flag

@ex.named_config
def fcnn():
    model_name = 'fcnn'  # Model name
    extra_fea_len = 128  # Extra feature length
    n_h = 2  # Number of hidden layers
    dropout_prob = 0.0  # Dropout probability

@ex.named_config
def att_fcnn():
    model_name = 'att_fcnn'  # Model name
    extra_fea_len = 128  # Extra feature length
    n_h = 2  # Number of hidden layers
    dropout_prob = 0.0  # Dropout probability
    task_att_type = 'self'  # Attention type: self or external
    att_S = 64  # S parameter of external attention
    

@ex.named_config
def tsd():
    data_dir = './data'  # Data directory
    batch_size = 128
    lr = 5e-3
    tasks = ['TSD']
    task_types = ['regression']

@ex.named_config
def tsd2():
    data_dir = './data'  # Data directory
    batch_size = 128
    lr = 5e-3
    tasks = ['TSD2']
    task_types = ['classification']
    

@ex.named_config
def ssd():
    data_dir = './data'  # Data directory
    batch_size = 128
    lr = 5e-3
    tasks = ['SSD']
    task_types = ['classification']

@ex.named_config
def tsd_ssd():
    data_dir = './data'  # Data directory
    batch_size = 32
    lr = 5e-3
    tasks = ['TSD', 'SSD']
    task_types = ['regression', 'classification']

@ex.named_config
def tsd2_ssd():
    data_dir = './data'  # Data directory
    batch_size = 32
    lr = 5e-3
    tasks = ['TSD2', 'SSD']
    task_types = ['classification', 'classification']
    
@ex.named_config
def ws24_water():
    data_dir = './data'  # Data directory
    batch_size = 256
    lr = 5e-3
    tasks = ['WS24_water']
    task_types = ['classification']

@ex.named_config
def ws24_water4():
    data_dir = './data'  # Data directory
    batch_size = 128
    lr = 5e-3
    tasks = ['WS24_water4']
    task_types = ['classification_4']

@ex.named_config
def ws24_acid():
    data_dir = './data'  # Data directory
    batch_size = 16
    lr = 5e-3
    tasks = ['WS24_acid']
    task_types = ['classification']

@ex.named_config
def ws24_base():
    data_dir = './data'  # Data directory
    batch_size = 16
    lr = 5e-3
    tasks = ['WS24_base']
    task_types = ['classification']

@ex.named_config
def ws24_boiling():
    data_dir = './data'  # Data directory
    batch_size = 16
    lr = 5e-3
    tasks = ['WS24_boiling']
    task_types = ['classification']

@ex.named_config
def tsd_ssd_ws24_water():
    data_dir = './data'  # Data directory
    batch_size = 32
    lr = 5e-3
    tasks = ['TSD', 'SSD', 'WS24_water']
    task_types = ['regression', 'classification', 'classification']
    dl_sampler = 'random'  # Data sampler type: random, same_task_in_batch
    loss_aggregation = 'fixed_weight_sum'  # Loss aggregation type: sum, trainable_weight_sum, sample_weight_sum, fixed_weight_sum
    task_weights = None

@ex.named_config
def tsd2_ssd_ws24_water():
    data_dir = './data'  # Data directory
    batch_size = 32
    lr = 5e-3
    tasks = ['TSD2', 'SSD', 'WS24_water']
    task_types = ['classification', 'classification', 'classification']
    dl_sampler = 'random'  # Data sampler type: random, same_task_in_batch
    loss_aggregation = 'fixed_weight_sum'  # Loss aggregation type: sum, trainable_weight_sum, sample_weight_sum, fixed_weight_sum
    task_weights = None

@ex.named_config
def tsd_ssd_ws24_water_water4():
    data_dir = './data'  # Data directory
    batch_size = 32
    lr = 5e-3
    tasks = ['TSD', 'SSD', 'WS24_water', 'WS24_water4']
    task_types = ['regression', 'classification', 'classification', 'classification_4']
    dl_sampler = 'random'  # Data sampler type: random, same_task_in_batch
    loss_aggregation = 'fixed_weight_sum'  # Loss aggregation type: sum, trainable_weight_sum, sample_weight_sum, fixed_weight_sum
    task_weights = None

@ex.named_config
def tsd2_ssd_ws24_water_water4():
    data_dir = './data'  # Data directory
    batch_size = 32
    lr = 5e-3
    tasks = ['TSD2', 'SSD', 'WS24_water', 'WS24_water4']
    task_types = ['classification', 'classification', 'classification', 'classification_4']
    dl_sampler = 'random'  # Data sampler type: random, same_task_in_batch
    loss_aggregation = 'fixed_weight_sum'  # Loss aggregation type: sum, trainable_weight_sum, sample_weight_sum, fixed_weight_sum
    task_weights = None

@ex.named_config
def tsd_ssd_ws24():
    data_dir = './data'  # Data directory
    batch_size = 16
    lr = 5e-3
    tasks = ['TSD', 'SSD', 'WS24_water', 'WS24_water4', 'WS24_acid', 'WS24_base', 'WS24_boiling']
    task_types = ['regression', 'classification', 'classification', 'classification_4', 'classification', 'classification', 'classification']
    dl_sampler = 'random' # Data sampler type: random, same_task_in_batch, same_ratio_prior
    loss_aggregation = 'fixed_weight_sum'  # Loss aggregation type: sum, trainable_weight_sum, sample_weight_sum, fixed_weight_sum
    # task_weights = [0.3, 0.25, 0.15, 0.15, 0.05, 0.05, 0.05]
    task_weights = None

@ex.named_config
def tsd2_ssd_ws24():
    data_dir = './data'  # Data directory
    batch_size = 16
    lr = 5e-3
    tasks = ['TSD2', 'SSD', 'WS24_water', 'WS24_water4', 'WS24_acid', 'WS24_base', 'WS24_boiling']
    task_types = ['classification', 'classification', 'classification', 'classification_4', 'classification', 'classification', 'classification']
    dl_sampler = 'random' # Data sampler type: random, same_task_in_batch, same_ratio_prior
    loss_aggregation = 'fixed_weight_sum'  # Loss aggregation type: sum, trainable_weight_sum, sample_weight_sum, fixed_weight_sum
    # task_weights = [0.3, 0.25, 0.15, 0.15, 0.05, 0.05, 0.05]
    task_weights = None

@ex.named_config
def ws24():
    data_dir = './data'  # Data directory
    batch_size = 32
    lr = 5e-3
    tasks = ['WS24_water', 'WS24_water4', 'WS24_acid', 'WS24_base', 'WS24_boiling']
    task_types = ['classification', 'classification_4', 'classification', 'classification', 'classification']
    dl_sampler = 'random' # Data sampler type: random, same_task_in_batch, same_ratio_prior
    loss_aggregation = 'fixed_weight_sum'  # Loss aggregation type: sum, trainable_weight_sum, sample_weight_sum, fixed_weight_sum
    task_weights = None

@ex.named_config
def ssd_ws24():
    data_dir = './data'  # Data directory
    batch_size = 32
    lr = 5e-3
    tasks = ['SSD', 'WS24_water', 'WS24_water4', 'WS24_acid', 'WS24_base', 'WS24_boiling']
    task_types = ['classification', 'classification', 'classification_4', 'classification', 'classification', 'classification']
    dl_sampler = 'random' # Data sampler type: random, same_task_in_batch, same_ratio_prior
    loss_aggregation = 'fixed_weight_sum'  # Loss aggregation type: sum, trainable_weight_sum, sample_weight_sum, fixed_weight_sum
    task_weights = None

# @ex.named_config
# def tsd_ssd_ws24():
#     data_dir = './data'  # Data directory
#     batch_size = 256
#     lr = 5e-3
#     tasks = ['TSD', 'SSD', 'WS24_water', 'WS24_water4']
#     task_types = ['regression', 'classification', 'classification', 'classification_4']   
    