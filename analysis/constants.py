IMG_OUTPUT_PATH = '/home/unix/vkrhk/EmmaEmb/EmmaEmb/img'
DATA_PATH = '/home/unix/vkrhk/EmmaEmb/data'
EMBEDDINGS_PATH = '/media/drive2/vkrhk/embeddings'
EMB_SPACES = [("ESM2", f"{EMBEDDINGS_PATH}/esm2_t33_650M_UR50D/layer_33/chopped_1022_overlap_300"),
("ANKH", f"{EMBEDDINGS_PATH}/ankh_base/layer_None/chopped_1022_overlap_300"),
("ProstT5", f"{EMBEDDINGS_PATH}/Rostlab/ProstT5/layer_None/chopped_1022_overlap_300/"),
("ProtT5", f"{EMBEDDINGS_PATH}/Rostlab/prot_t5_xl_uniref50/layer_None/chopped_1022_overlap_300"),
("ESM1", f"{EMBEDDINGS_PATH}/esm1_t34_670M_UR100/layer_34/chopped_1022_overlap_300/"),
("ESMC", f"{EMBEDDINGS_PATH}/esmc-300m-2024-12/layer_None/chopped_1022_overlap_300/"),]

CRYPTOBENCH_TRAIN_DATASET = f'{DATA_PATH}/train.txt'
CRYPTOBENCH_TEST_DATASET = f'{DATA_PATH}/test.txt'
SCPDB_DATASET = f'{DATA_PATH}/scPDB_filtered.csv' # controled sequence similarity (< 10 % with LIGYSIS dataset)
LIGYSIS_DATASET = f'{DATA_PATH}/ligysis_for_residue_level_evaluation.csv'