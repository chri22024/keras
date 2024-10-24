import os
import sys


DST_DIR = 'D01_estimator'
SRC_DIR = 'D00_dataset/labeled_training'
EST_FILE = os.path.join(TRN_DST_DIR, 'estimator.h5')
INFO_FILE = os.path.join(TRN_DST_DIR, 'model_info.txt')
GRAPH_FILE = os.path.join(TRN_DST_DIR, 'model_graph.pdf')
HIST_FILE = os.path.join(TRN_DST_DIR, 'history.pdf')
DENSE_DIMS = [4096, 2048, 1024, 128]
LR = 1e-3
MIN_LR = 1e-7
BATCH_SIZE = 32
EPOCHS = 10
VALID_RATE = 0.2
ES_PATIENCE = 30
LR_PATIENCE = 10



from P01_model_maker import ModelMaker
maker = ModelMaker(
    src_dir = SRC_DIR,
    dst_dir = DST_DIR,
    est_file = EST_FILE,
    info_file = INFO_FILE,
    graph_file = GRAPH_FILE,
    hist_file = HIST_FILE,
    dense_dims =DENSE_DIMS,
    lr = LR,
    min_le = MIN_LR,
    batch_size = TRN_BATCH_SIZE,
    epochs = TRN_EPOCHS,
    vaild_rate = TRN_VALID_RATE,
    es_patience = ES_PATIENCE,
    lr_patience = LR_PATIENCE
)

maker.execute()
