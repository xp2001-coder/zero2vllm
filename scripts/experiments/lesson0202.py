"""Pinned public checkpoint -> our CPU decoder -> Transformers comparison.

Network is used only with --download. All inference uses the local snapshot.
No access token, remote Python code or model weights are stored in reports.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.common import source_fingerprint

LESSON = ROOT/'docs/chapters/part-02-engine/02-loading-real-model-weights'
RECIPE = LESSON/'data/model.json'


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def prepare_snapshot(directory, recipe, download=False):
    from zero2vllm.modeling.checkpoint import read_json, sha256
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=recipe['repo_id'], revision=recipe['revision'],
                          allow_patterns=recipe['files'], local_dir=directory, token=False)
        files = [{'file': name, 'sha256': sha256(directory/name), 'bytes': (directory/name).stat().st_size}
                 for name in sorted(recipe['files'])]
        save_json(directory/'zero2vllm-snapshot.json', {
            'repo_id': recipe['repo_id'], 'revision': recipe['revision'], 'files': files,
        })
    receipt_path = directory/'zero2vllm-snapshot.json'
    if not receipt_path.is_file():
        raise ValueError('No snapshot receipt. Run once with --download; later runs are offline.')
    receipt = read_json(receipt_path)
    if (receipt.get('repo_id'), receipt.get('revision')) != (recipe['repo_id'], recipe['revision']):
        raise ValueError('Snapshot repo/revision differs from the lesson recipe.')
    entries = receipt.get('files', [])
    if sorted(item['file'] for item in entries) != sorted(recipe['files']):
        raise ValueError('Snapshot receipt file list differs from the recipe.')
    for item in entries:
        path = directory/item['file']
        if not path.is_file() or sha256(path) != item['sha256']:
            raise ValueError(f'Snapshot file missing or changed: {item["file"]}')
    return receipt


def comparison(name, actual, expected, atol, rtol):
    import torch
    same_shape = actual.shape == expected.shape
    finite = same_shape and bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    diff = (actual-expected).abs() if finite else None
    return {'stage': name, 'shape': list(actual.shape),
            'max_abs_error': diff.max().item() if finite else None,
            'mean_abs_error': diff.mean().item() if finite else None,
            'passed': finite and torch.allclose(actual, expected, atol=atol, rtol=rtol)}


def compare_models(model, reference, tokenizer, prompts, max_new_tokens, atol, rtol):
    import torch
    from zero2vllm.modeling.llama import greedy_generate
    results = []
    with torch.inference_mode():
        for prompt in prompts:
            # One tokenization is shared by both paths; no implicit BOS or chat template.
            tokens = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids']
            if not 1 <= tokens.shape[1] <= 128:
                raise ValueError('Use 1..128 input tokens for this dense CPU verification experiment.')
            trace = {}
            def capture(name):
                def hook(module, inputs, output):
                    trace[name] = output[0] if isinstance(output, tuple) else output
                return hook
            handles = [reference.model.embed_tokens.register_forward_hook(capture('embedding')),
                       reference.model.norm.register_forward_hook(capture('final_norm'))]
            handles += [layer.register_forward_hook(capture(f'layers.{i}'))
                        for i, layer in enumerate(reference.model.layers)]
            try:
                expected = reference(tokens, use_cache=False).logits
            finally:
                for handle in handles:
                    handle.remove()
            actual, ours = model(tokens, return_trace=True)
            stages = [comparison(name, ours[name], trace[name], atol, rtol) for name in ours]
            stages.append(comparison('logits', actual, expected, atol, rtol))
            ours_ids = greedy_generate(model, tokens, max_new_tokens, tokenizer.eos_token_id)
            # Transformers' own generation loop, with all relevant settings explicit.
            from transformers import GenerationConfig
            generation = GenerationConfig(do_sample=False, num_beams=1, use_cache=False,
                                          eos_token_id=tokenizer.eos_token_id,
                                          pad_token_id=tokenizer.eos_token_id,
                                          bos_token_id=tokenizer.bos_token_id)
            if max_new_tokens:
                expected_sequence = reference.generate(tokens, attention_mask=torch.ones_like(tokens),
                                                       generation_config=generation, max_new_tokens=max_new_tokens)
                expected_ids = expected_sequence[0, tokens.shape[1]:].tolist()
            else:
                expected_ids = []
            scores, ids = actual[0, -1].topk(5)
            results.append({
                'prompt': prompt, 'input_ids': tokens[0].tolist(), 'input_tokens': tokens.shape[1],
                'comparisons': stages,
                'last_top5': [{'id': token, 'logit': score} for token, score in zip(ids.tolist(), scores.tolist())],
                'our_generated_ids': ours_ids, 'reference_generated_ids': expected_ids,
                'our_completion': tokenizer.decode(ours_ids, skip_special_tokens=True),
                'reference_completion': tokenizer.decode(expected_ids, skip_special_tokens=True),
                'all_stages_close': all(stage['passed'] for stage in stages),
                'greedy_ids_equal': ours_ids == expected_ids,
                'passed': all(stage['passed'] for stage in stages) and ours_ids == expected_ids,
            })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('.local/models/smollm2-135m'))
    parser.add_argument('--download', action='store_true', help='Download the pinned public snapshot once.')
    parser.add_argument('--inspect-only', action='store_true', help='Audit weights without running the oracle.')
    parser.add_argument('--output-dir', type=Path, default=Path('.local/runs/part-02-engine/02-loading-real-model-weights'))
    parser.add_argument('--prompt', action='append', help='Override the three default prompts; repeatable.')
    parser.add_argument('--max-new-tokens', type=int, default=8)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or not 0 <= args.max_new_tokens <= 32:
        parser.error('Require threads >= 1 and max-new-tokens in 0..32.')
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from zero2vllm.modeling.checkpoint import load_checkpoint, read_json
    except ModuleNotFoundError as error:
        parser.exit(2, f'Missing optional dependency {error.name}. Install the checkpoint extra as described in lesson 2.2.\n')
    torch.set_num_threads(args.threads)
    directory, output = (ROOT/args.snapshot).resolve(), (ROOT/args.output_dir).resolve()
    recipe = read_json(RECIPE)
    print('Checking snapshot...', flush=True)
    receipt = prepare_snapshot(directory, recipe, args.download)
    model, audit = load_checkpoint(directory)
    save_json(output/'weights.json', audit)
    print(f'Loaded {audit["tensor_count"]} tensors; {audit["parameter_count"]:,} parameters.', flush=True)
    summary = {
        'experiment': 'part2-02-real-checkpoint', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': source_fingerprint(), 'checkpoint': receipt,
        'environment': {'python': platform.python_version(), 'system': platform.platform(),
                        **{name: importlib.metadata.version(name) for name in
                           ('torch', 'numpy', 'transformers', 'safetensors', 'huggingface-hub', 'tokenizers')},
                        'device': 'cpu', 'dtype': 'float32', 'threads': args.threads},
        'config': asdict(model.config), 'parameter_count': audit['parameter_count'],
        'atol': recipe['atol'], 'rtol': recipe['rtol'], 'attention': 'eager', 'use_cache': False,
        'add_special_tokens': False, 'chat_template': False, 'max_new_tokens': args.max_new_tokens,
        'gpu_status': {'cuda-a100': 'not_run', 'maca-c500': 'not_run'},
        'weights_output': 'weights.json', 'verification': 'not_run', 'passed': None,
    }
    if not args.inspect_only:
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        # Added tokens can make len(tokenizer) differ from tokenizer.vocab_size.
        if len(tokenizer) > model.config.vocab_size:
            raise ValueError('Tokenizer contains IDs outside the model vocabulary.')
        reference = AutoModelForCausalLM.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32, attn_implementation='eager', use_safetensors=True,
        ).eval()
        cases = compare_models(model, reference, tokenizer, args.prompt or recipe['prompts'],
                               args.max_new_tokens, recipe['atol'], recipe['rtol'])
        summary.update({'verification': 'executed', 'cases': cases, 'passed': all(c['passed'] for c in cases),
                        'tokenizer': {'class': type(tokenizer).__name__, 'length': len(tokenizer),
                                      'bos_token_id': tokenizer.bos_token_id, 'eos_token_id': tokenizer.eos_token_id}})
        for case in cases:
            print(f'{"PASS" if case["passed"] else "FAIL"} {case["prompt"]!r}: {case["our_generated_ids"]}', flush=True)
    save_json(output/'summary.json', summary)
    print(f'Summary: {output/"summary.json"}', flush=True)
    return 1 if summary['passed'] is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
