import json
from tiktoken import Encoding
from tiktoken.load import data_gym_to_mergeable_bpe_ranks

tokenizer_config = "tokenizer.json"


def data_gym_to_mergeable_bpe_ranks(bpe_merges, start_idx=0):
  # NB: do not add caching to this function
  rank_to_intbyte = [b for b in range(2**8) if chr(b).isprintable() and chr(b) != " "]

  data_gym_byte_to_byte = {chr(b): b for b in rank_to_intbyte}
  n = 0
  for b in range(2**8):
    if b not in rank_to_intbyte:
      rank_to_intbyte.append(b)
      data_gym_byte_to_byte[chr(2**8 + n)] = b
      n += 1
  assert len(rank_to_intbyte) == 2**8
  bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_merges]

  def decode_data_gym(value: str) -> bytes:
    return bytes(data_gym_byte_to_byte[b] for b in value)

  # add the single byte tokens
  bpe_ranks = {bytes([b]): i + start_idx for i, b in enumerate(rank_to_intbyte)}
  # add the merged tokens
  n = len(bpe_ranks) + start_idx
  for first, second in bpe_merges:
    bpe_ranks[decode_data_gym(first) + decode_data_gym(second)] = n
    n += 1
  return bpe_ranks


def create_tokenizer():
  special_tokens = [
    "[PAD]", "<|endoftext|>", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<|beginoftext|>", "<|fim_prefix|>",
    "<|fim_middle|>", "<|fim_suffix|>", "<|beginofprompt|>", "<|endofprompt|>", "[PAD1]", "[PAD2]", "[PAD3]", "[PAD4]",
    "[PAD5]", "[PAD6]", "[PAD7]", "[PAD8]", "[PAD9]", "[PAD10]", "[PAD11]", "[PAD12]", "[PAD13]", "[PAD14]", "[PAD15]",
    "[PAD16]"
  ]

  cjk_regex = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}--[\u4e00-\u9fa5]]+|[\u4e00-\u9fa5]|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?![\S--[\u4e00-\u9fa5]])|\s+"""
  with open(tokenizer_config) as fr:
    ddict = json.loads(fr.read())
    bpe_ranks = data_gym_to_mergeable_bpe_ranks(ddict["model"]["merges"], start_idx=len(special_tokens))
    sp_ranks = {t: i for i, t in enumerate(special_tokens)}
    targs = {"name": "cjk100k", "pat_str": cjk_regex, "mergeable_ranks": bpe_ranks, "special_tokens": sp_ranks}
  return Encoding(**targs)


batch_docs = ["你好，世界", "hello, world!"]
tokenizer = create_tokenizer()
tokenized_docs = tokenizer.encode_batch(batch_docs, allowed_special="all", num_threads=8)
recovered = tokenizer.decode_batch(tokenized_docs)

print(tokenized_docs, recovered)