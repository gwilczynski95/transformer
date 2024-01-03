from importlib.machinery import SourceFileLoader
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import create_masks
from translate import load_model_and_set


def _get_model_inputs(loader, device):
    val_set = loader.get_set(1, "valid", shuffle=True)
    src, tgt, src_lens, tgt_lens = next(iter(val_set))
    src = src.to(device).T
    tgt = tgt.to(device).T
    tgt_input = tgt[:, :-1]
    return src, tgt_input, src_lens, tgt_lens


def parse_tokens(tokens, vocab_transform):
    out = []
    for sent_idx in range(tokens.shape[0]):
        sentence = vocab_transform.lookup_tokens(
            tokens[sent_idx, :].cpu().numpy().tolist()
        )[1:]
        out.append(["<bos>"] + sentence)
    return out


def visualize(config, model, mode, x_ticks, x_tokens, y_ticks, y_tokens):
    assert mode in ["enc", "dec_1", "dec_2"]

    if mode == "enc":
        fig_name = "Encoder attention"
    elif mode == "dec_1":
        fig_name = "Decoder first attention"
    else:
        fig_name = "Decoder second attention"

    fig = plt.figure(constrained_layout=True)
    fig.suptitle(fig_name)
    subfigs = fig.subfigures(
        nrows=config.model_hyperparams["enc_dec_blocks"],
        ncols=1
    )
    for layer_idx in range(config.model_hyperparams["enc_dec_blocks"]):
        subfig = subfigs[layer_idx]
        subfig.suptitle(f"Layer {layer_idx}")
        axs = subfig.subplots(
            nrows=1,
            ncols=config.model_hyperparams["attention_heads"]
        )

        if mode == "enc":
            attn_vals = model.encoder.enc_blocks[layer_idx].mha.attn_val.detach().cpu().numpy()[0]
        elif mode == "dec_1":
            attn_vals = model.decoder.dec_blocks[layer_idx].d_mha.attn_val.detach().cpu().numpy()[0]
        else:
            attn_vals = model.decoder.dec_blocks[layer_idx].mha.attn_val.detach().cpu().numpy()[0]

        for attn_head_idx in range(config.model_hyperparams["attention_heads"]):
            head_attn_vals = attn_vals[attn_head_idx]
            axs[attn_head_idx].matshow(head_attn_vals, cmap=plt.cm.Blues, aspect="auto")
            axs[attn_head_idx].xaxis.set_ticks_position("bottom")
            if not layer_idx:
                axs[attn_head_idx].set_title(f"Head {attn_head_idx}")
            if attn_head_idx:
                axs[attn_head_idx].set_yticks([])
            else:
                axs[attn_head_idx].set_yticks(y_ticks)
                axs[attn_head_idx].set_yticklabels(y_tokens, fontsize=10)
            axs[attn_head_idx].set_xticks(x_ticks)
            axs[attn_head_idx].set_xticklabels(x_tokens, fontsize=10, rotation="vertical")
    plt.show()


def main(model_path, config):
    model, loader = load_model_and_set(model_path, config)
    src, tgt_input, src_lens, tgt_lens = _get_model_inputs(loader, model.device)

    src_tokens = parse_tokens(src, loader.vocab_transform["de"])[0]
    tgt_input_tokens = parse_tokens(tgt_input, loader.vocab_transform["en"])[0]

    src_mask, tgt_mask = create_masks(src, tgt_input, loader.pad_idx)

    model.eval()
    model(src, tgt_input, src_lens, tgt_lens, src_mask, tgt_mask)

    src_ticks = np.arange(len(src_tokens))
    tgt_ticks = np.arange(len(tgt_input_tokens))

    # mode = "enc"
    #
    # visualize(
    #     config,
    #     model,
    #     mode,
    #     src_ticks,
    #     src_tokens,
    #     src_ticks,
    #     src_tokens,
    # )

    # mode = "dec_1"
    #
    # visualize(
    #     config,
    #     model,
    #     mode,
    #     tgt_ticks,
    #     tgt_input_tokens,
    #     tgt_ticks,
    #     tgt_input_tokens,
    # )

    mode = "dec_2"

    visualize(
        config,
        model,
        mode,
        src_ticks,
        src_tokens,
        tgt_ticks,
        tgt_input_tokens,
    )


if __name__ == '__main__':
    path_to_model = ""

    path_to_config = Path(Path(path_to_model).parent.parent, "config.py")
    model_config = SourceFileLoader("config", str(path_to_config)).load_module()
    main(path_to_model, model_config)
