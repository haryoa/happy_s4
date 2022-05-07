"""
Database of data 
"""


DATA_INP_LIST = {
    "super_glue/rte": {
        "data_name": "super_glue",
        "data_subset": "rte",
        "column_train": ["premise", "hypothesis"],
        "column_label": "label",
    },
    "super_glue/boolq": {
        "data_name": "super_glue",
        "data_subset": "boolq",
        "column_train": ["passage", "question"],
        "column_label": "label",
    },
    "super_glue/cb": {
        "data_name": "super_glue",
        "data_subset": "cb",
        "column_train": ["premise", "hypothesis"],
        "column_label": "label",
    },
    "super_glue/copa": {
        "data_name": "super_glue",
        "data_subset": "copa",
        "column_train": ["premise", "hypothesis"],
        "column_label": "label",
    },
    "super_glue/multirc": {
        "data_name": "super_glue",
        "data_subset": "multirc",
        "column_train": ["paragraph", "question", "answer"],
        "column_label": "label",
    },
    "glue/mrpc": {
        "data_name": "glue",
        "data_subset": "mrpc",
        "column_train": ["sentence1", "sentence2"],
        "column_label": "label",
    },
    "glue/sst2": {
        "data_name": "glue",
        "data_subset": "sst2",
        "column_train": ["sentence"],
        "column_label": "label",
    },
    "glue/qqp": {
        "data_name": "glue",
        "data_subset": "qqp",
        "column_train": ["question1", "question2"],
        "column_label": "label",
    },
    "glue/qnli": {
        "data_name": "glue",
        "data_subset": "qnli",
        "column_train": ["question", "sentence"],
        "column_label": "label",
    },
    "glue/rte": {
        "data_name": "glue",
        "data_subset": "rte",
        "column_train": ["sentence1", "sentence2"],
        "column_label": "label",
    },
    "glue/wnli": {
        "data_name": "glue",
        "data_subset": "wnli",
        "column_train": ["sentence1", "sentence2"],
        "column_label": "label",
    },
}
