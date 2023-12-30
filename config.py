experiment_name = "adam_test_1"
experiment_parent_path = ""
checkpoint_path = None
batch_size = 64

_multi_30k_train_samples = 29008

model_hyperparams = {
    "enc_dec_blocks": 6,
    "model_dim": 512,
    "attention_heads": 8,
    "pwff_mid_dim": 2048,
    "dropout_rate": 0.1
}

training_hyperparams = {
    "epochs": int(105000 / (_multi_30k_train_samples / batch_size)),
    "optimizer_params": {
        "lr": 1,
        "betas": (0.9, 0.98),
        "eps": 1e-9,
        "warmup_steps": 4000,
        "d_model": model_hyperparams["model_dim"],
    },
    "checkpoint_path": checkpoint_path
}
