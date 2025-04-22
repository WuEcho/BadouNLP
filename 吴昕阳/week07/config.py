# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "./text_categories_test/text_categories_test_info.csv",
    "valid_data_path": "./text_categories_test/text_categories_test_info.csv",
    "vocab_path":"chars.txt",
    "model_type":"",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3, #学习率 如果是bert模型，建议调小一点设置为1e-5
    "pretrain_model_path":r"./bert-base-chinese", #预训练模型路径
    "seed": 987
}

