## My very own Transformer fun in Pytorch 😊

In this repo I try to implement the Transformer from the famous 
[Attention is All You Need paper](https://arxiv.org/pdf/1706.03762.pdf).

It's task will be, of course, the translation task.

### Run

To run this project you have to setup hyperparams in `config.py`. Then you 
just run `run.py`!

#### My notes

My first implementation of Scaled Dot Product Attention and Multi Head Attention was extremely unproficient. This is 
where I supplemented my raw knowledge (from the paper) with this 
[Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/). The result was quite satisfying.

For those params in config file I managed to create a simple Transformer that can do some translating:

```python
checkpoint_path = None
batch_size = 64

_multi_30k_train_samples = 29008

model_hyperparams = {
    "enc_dec_blocks": 3,
    "model_dim": 128,
    "attention_heads": 4,
    "pwff_mid_dim": 256,
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
    "loss": {
        "smoothing_factor": 0.1
    },
    "checkpoint_path": checkpoint_path
}
```

And then I had (with greedy sampling) fun with translations:

```
Type the sentence: Ein Mann hält eine Waffe .  # A man is holding a gun
A man is holding a gun .

Type the sentence: Ich bin 32 Jahre alt und Arzt .  # I am 32 years old and I am a doctor
I am say hold blocked and fake doctor .

Type the sentence: Ich mag Brot mit Butter sehr .  # I really like bread with butter
I looking at bread stand with bread .

Type the sentence: Ich mag mein Auto .  # I like my car
I am looking my car .
```

So as you can see - sometimes it's right, other times it's completely wrong, but still it rather holds an idea of 
the input.

It was fun.