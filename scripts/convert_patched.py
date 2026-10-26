#!/usr/bin/env python3
"""convert_hf_to_gguf wrapper: neutralizes the transformers 5.20
LlamaConfig divisibility validator (rejects 18 heads x head_dim 128
even though the architecture is valid GQA), then runs the stock
convert script. Usage: convert_patched.py HF_DIR OUTFILE"""
import runpy, sys, os
import transformers.models.llama.configuration_llama as lc
lc.LlamaConfig.validate_architecture = lambda self: None
import transformers.configuration_utils as cu
cu.PretrainedConfig.validate = lambda self: None  # registry holds fns by ref; kill the entry point
try:
    import huggingface_hub.dataclasses as hd
    hd.validate = lambda obj, **k: None
except Exception:
    pass
LC = os.path.expanduser("~/workspace/publication-gap-builds/track6-synth-replication/llama.cpp")
sys.path.insert(0, LC)  # convert script imports sibling package 'conversion'
sys.argv = ["convert_hf_to_gguf.py", sys.argv[1], "--outtype", "q8_0", "--outfile", sys.argv[2]]
runpy.run_path(os.path.join(LC, "convert_hf_to_gguf.py"), run_name="__main__")
