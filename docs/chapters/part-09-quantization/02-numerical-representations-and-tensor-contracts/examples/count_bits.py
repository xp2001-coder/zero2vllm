"""Account for each declared storage component, including padding."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
from scripts.experiments.lesson0902 import load_fixture
from zero2vllm.quantization.tensor_contract import quantize_asymmetric,calculate_effective_bpw

if __name__=='__main__':
    t=quantize_asymmetric(load_fixture()['w'],scale_dtype='float16',pad_to_group=True)
    print(json.dumps(calculate_effective_bpw(t,metadata_bytes=4,alignment_bytes=16),indent=2))
    print('48 estimated bytes / 15 real weights = 25.6 bits per weight. No checkpoint is written.')
