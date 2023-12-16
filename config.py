experiment_name = "test_1"
checkpoint_path = None

_batch_size = 64
_multi_30k_train_samples = 29008

model_hyperparams = {
    "enc_dec_blocks": 6,
    "model_dim": 512,
    "attention_heads": 8,
    "pwff_mid_dim": 2048,
    "dropout_rate": 0.1
}

training_hyperparams = {
    "batch_size": _batch_size,
    "epochs": int(105000 / (_multi_30k_train_samples / _batch_size)),
    "optimizer": {
        "warmup_steps": 4000,
        "beta_1": 0.9,
        "beta_2": 0.98,
        "eps": 1e-9
    }
}
