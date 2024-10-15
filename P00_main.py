import os



SRC_DIR = 'D00_dataset/training'
DST_DIR = 'D01_estimator'
EST_FILE = os.path.join(DST_DIR, 'estimator.h5')
INFO_FILE = os.path.join(DST_DIR, 'model_info.txt')
GRAPH_FILE = os.path.join(DST_DIR, 'model_graph.pdf')
HIST_FILE = os.path.join(DST_DIR, 'history.pdf')
INPUT_SIEZ = (160, 160)
FILTERS = (64, 128, 256, 512, 1024)
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DENSE_DIMS = [4096, 2048, 1024, 128]
LR = 1e-3
MIN_LR = 1e-7
BATCH_SIZE = 32
EPOCHS = 10
VALID_RATE = 0.2
REUSE_CNT = 1
ES_PATIENCE = 30
LR_PATIENCE = 10
n_class = len(os.listdir(SRC_DIR))
DENSE_DIMS.append(n_class)

from P01_model_maker import ModelMaker
maker = ModelMaker(
    src_dir = SRC_DIR,
    dst_dir = DST_DIR,
    est_file = EST_FILE,
    info_file = INFO_FILE,
    graph_file = GRAPH_FILE,
    input_size = INPUT_SIEZ,
    hist_file = HIST_FILE,
    filters = FILTERS,
    kernel_size = KERNEL_SIZE,
    pool_size = POOL_SIZE,
    dense_dims = DENSE_DIMS,
    lr = LR,
    min_lr = MIN_LR,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    vaild_rate = VALID_RATE,
    reuse_cnt = REUSE_CNT,
    es_patience = ES_PATIENCE,
    lr_patience = LR_PATIENCE


)

maker.execute()

