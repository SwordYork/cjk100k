import json

from tiktoken.load import load_tiktoken_bpe
from tokenizers import decoders


def load_tiktoken_cl100k():
  """Load a TikToken BPE from a URL."""
  decoder = decoders.ByteLevel()
  mergeable_ranks = load_tiktoken_bpe("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken")

  # copy from https://github.com/openai/tiktoken/blob/main/tiktoken/load.py
  rank_to_intbyte = [b for b in range(2**8) if chr(b).isprintable() and chr(b) != " "]
  data_gym_byte_to_byte = {chr(b): b for b in rank_to_intbyte}
  n = 0
  for b in range(2**8):
    if b not in rank_to_intbyte:
      rank_to_intbyte.append(b)
      data_gym_byte_to_byte[chr(2**8 + n)] = b
      n += 1
  assert len(rank_to_intbyte) == 2**8
  int_to_gym = {v: k for k, v in data_gym_byte_to_byte.items()}

  def encode_data_gym(value: bytes) -> str:
    return "".join(int_to_gym[byte] for byte in value)

  id2token = {}
  for idx, k in enumerate(mergeable_ranks):
    # convert to str token from bytes by ByteLevel decoder
    d = decoder.decode([encode_data_gym(k)])
    id2token[idx] = d
  return id2token


def load_cjk100k():
  decoder = decoders.ByteLevel()
  ddict = json.loads(open("tokenizer.json").read())
  id2token = {}
  for idx, v in enumerate(ddict["model"]["vocab"]):
    d = decoder.decode([v])
    id2token[idx] = d
  return id2token