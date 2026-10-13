"""Define shared token identities and supported model architectures."""

from esm.data import Alphabet

ALPHABET = Alphabet.from_architecture("ESM-1b")
BATCH_CONVERTER = ALPHABET.get_batch_converter()
CLS_IDX = ALPHABET.cls_idx
EOS_IDX = ALPHABET.eos_idx
PAD_IDX = ALPHABET.padding_idx
MASK_IDX = ALPHABET.mask_idx
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_IDS = [ALPHABET.tok_to_idx[aa] for aa in AA20]
SEQUENCE_ORDER = ("HLA_a", "HLA_b", "Peptide", "TCR_a", "TCR_b")
SEQUENCE_CONVENTIONS = ("v1", "v2")
SCORING_RECIPE = "masked-peptide-full-vocabulary-v1"

ESM2_ARCH = {
    "ESM2_8M": {"num_layers": 6, "embed_dim": 320, "attention_heads": 20},
    "ESM2_35M": {"num_layers": 12, "embed_dim": 480, "attention_heads": 20},
    "ESM2_150M": {"num_layers": 30, "embed_dim": 640, "attention_heads": 20},
    "ESM2_650M": {"num_layers": 33, "embed_dim": 1280, "attention_heads": 20},
    "ESM2_3B": {"num_layers": 36, "embed_dim": 2560, "attention_heads": 40},
    "ESM2_15B": {"num_layers": 48, "embed_dim": 5120, "attention_heads": 40},
}

ESMC_ARCH = {
    "DecoderTCRC_300M": {"d_model": 960, "n_heads": 15, "n_layers": 30},
    "DecoderTCRC_600M": {"d_model": 1152, "n_heads": 18, "n_layers": 36},
    "DecoderTCRC_6B": {"d_model": 2560, "n_heads": 40, "n_layers": 80},
}
