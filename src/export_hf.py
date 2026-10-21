"""Export the learned model as GPT-2 without changing its weights."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from src.pipeline import load_checkpoint, write_json, file_hash

def convert(model, config, tokenizer):
    from transformers import GPT2Config, GPT2LMHeadModel
    settings = GPT2Config(vocab_size=config["vocab_size"], n_positions=config["context_length"],
                          n_ctx=config["context_length"], n_embd=config["emb_dim"],
                          n_layer=config["n_layers"], n_head=config["n_heads"],
                          activation_function="gelu_new", layer_norm_epsilon=1e-5,
                          resid_pdrop=config["drop_rate"], embd_pdrop=config["drop_rate"],
                          attn_pdrop=config["drop_rate"], tie_word_embeddings=False,
                          bos_token_id=tokenizer.game_start_id, eos_token_id=tokenizer.game_end_id)
    settings._attn_implementation = "eager"
    target = GPT2LMHeadModel(settings).eval()
    source = model.state_dict()
    weights = target.state_dict()
    weights["transformer.wte.weight"] = source["tok_emb.weight"]
    weights["transformer.wpe.weight"] = source["pos_emb.weight"]
    weights["lm_head.weight"] = source["out_head.weight"]
    for layer in range(config["n_layers"]):
        src, dst = f"trf_blocks.{layer}.", f"transformer.h.{layer}."
        for ours, theirs in (("norm1", "ln_1"), ("norm2", "ln_2")):
            weights[dst + theirs + ".weight"] = source[src + ours + ".scale"]
            weights[dst + theirs + ".bias"] = source[src + ours + ".shift"]
        weights[dst + "attn.c_attn.weight"] = torch.cat([source[src + "att.W_" + name + ".weight"].T for name in ("query", "key", "value")], dim=1)
        weights[dst + "attn.c_attn.bias"] = torch.cat([source.get(src + "att.W_" + name + ".bias", torch.zeros(config["emb_dim"])) for name in ("query", "key", "value")])
        weights[dst + "attn.c_proj.weight"] = source[src + "att.out_proj.weight"].T
        weights[dst + "attn.c_proj.bias"] = source[src + "att.out_proj.bias"]
        for ours, theirs in (("ff.layers.0", "mlp.c_fc"), ("ff.layers.2", "mlp.c_proj")):
            weights[dst + theirs + ".weight"] = source[src + ours + ".weight"].T
            weights[dst + theirs + ".bias"] = source[src + ours + ".bias"]
    weights["transformer.ln_f.weight"] = source["final_norm.scale"]
    weights["transformer.ln_f.bias"] = source["final_norm.shift"]
    target.load_state_dict(weights)
    return target

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    model, tokenizer, checkpoint = load_checkpoint(a.checkpoint)
    exported = convert(model, checkpoint["config"], tokenizer)
    # Verify conversion against several independent token sequences.
    generator = torch.Generator().manual_seed(19)
    max_error = 0.0
    with torch.no_grad():
        for length in [1, min(17, model.pos_emb.num_embeddings), model.pos_emb.num_embeddings]:
            ids = torch.randint(tokenizer.vocab_size, (2, length), generator=generator)
            original = model(ids)
            result = exported(ids, use_cache=False).logits
            torch.testing.assert_close(original, result, rtol=2e-4, atol=2e-5)
            max_error = max(max_error, (original - result).abs().max().item())
    exported.save_pretrained(a.output)
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(models.WordLevel(vocab=tokenizer.move_to_id, unk_token="<|game_end|>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<|game_start|>",
                                           eos_token="<|game_end|>", unk_token="<|game_end|>", pad_token="<|game_end|>")
    hf_tokenizer.save_pretrained(a.output)
    write_json(Path(a.output) / "chess_metadata.json",
               {"checkpoint_sha256": file_hash(a.checkpoint), "max_logit_error": max_error,
                "board_state": hasattr(tokenizer, "board_ids"), "vocabulary": tokenizer.move_to_id})
    print("Verified GPT-2 export:", a.output, "| Max logit error:", max_error)

if __name__ == "__main__":
    main()
