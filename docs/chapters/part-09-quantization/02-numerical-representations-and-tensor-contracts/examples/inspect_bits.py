"""Interpret a bit pattern, then encode a finite value in a chosen format."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
from zero2vllm.quantization.minifloat import decode_minifloat,encode_minifloat

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format',choices=('e4m3fn','e5m2','e2m1'),default='e4m3fn')
    parser.add_argument('--code',type=lambda x:int(x,0),default=0x3d)
    parser.add_argument('--value',type=float,default=1.625)
    parser.add_argument('--ftz',action='store_true')
    args=parser.parse_args()
    try:
        decoded=decode_minifloat(args.code,args.format,ftz=args.ftz)
        encoded=encode_minifloat(args.value,args.format)
    except ValueError as e: parser.error(str(e))
    print('bit pattern:',hex(args.code),'format:',args.format,'decoded:',decoded)
    print('input:',args.value,'encoded:',hex(encoded),'restored:',decode_minifloat(encoded,args.format))
    print('Finite-input nearest-even encoder, saturating overflow. No native low-precision arithmetic.')
