"""
HunyuanWorld-Mirror real inference runner.
==========================================
Runs GENUINE inference against Tencent's official HunyuanWorld-Mirror Gradio
Space and downloads the real 3D outputs (3DGS / point cloud / depth / normal /
camera params). Nothing here is fabricated or CSS-simulated.

WHY A HF TOKEN IS REQUIRED
--------------------------
The Space runs on HuggingFace ZeroGPU. Anonymous traffic shares a small daily
quota (~120-300 GPU-seconds). One reconstruction needs ~180 GPU-seconds, so an
anonymous run is usually rejected with:
    "You have exceeded your ZeroGPU quota (180s requested vs. 123s left)"
A free HuggingFace account token raises the quota substantially. Register at
https://huggingface.co/join then create a READ token at
https://huggingface.co/settings/tokens (free, no credit card).

USAGE
-----
    # Recommended: log in once (token saved to ~/.cache/huggingface/token),
    # then run with NO token on the command line:
    #   hf auth login          # paste token at the prompt
    #   python real_run/run_inference.py
    # The script auto-reads the cached token via huggingface_hub.get_token().

    # Or pass an env var
    set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
    python real_run/run_inference.py

    # Or pass inline (no env var needed)
    python real_run/run_inference.py --token hf_xxxxx

    # Use your own image
    python real_run/run_inference.py --token hf_xxxxx --image path/to/your.jpg

    # Pick a different mirror if the official Space is busy
    python real_run/run_inference.py --token hf_xxxxx \
        --space https://paulhouston-hunyuanworld-mirror.hf.space/

NOTES
-----
- Input defaults to an official example image shipped in the Space repo.
- The sandbox HTTP proxy breaks gradio_client (502); the script clears proxy
  env vars and connects directly, which was verified to work.
- Every run writes a run_meta.json with source_type="real_model", the endpoint,
  model version, timing, and every downloaded artifact + its byte size.
  This metadata is designed to be imported into the eval bench as real evidence.
"""
import os, sys, json, time, argparse, traceback

# gradio_client's httpx chokes on the sandbox proxy (502) -> connect directly.
for _k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(_k, None)

from gradio_client import Client, handle_file

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPACE = 'https://tencent-hunyuanworld-mirror.hf.space/'
DEFAULT_IMG = os.path.join(HERE, 'input', 'bright_room_1.jpg')
OUT_DIR = os.path.join(HERE, 'output')
META_PATH = os.path.join(HERE, 'run_meta.json')

OUTPUT_LABELS = [
    'pointmap_mesh', 'value_17', 'frame', 'depth_map', 'normal_map',
    'slider1', 'slider2', 'html1', 'html2', 'camera_params',
    'gaussian_splatting', 'rgb_video', 'depth_video', 'terminal',
]
FILE_OUTPUTS = {'pointmap_mesh', 'depth_map', 'normal_map', 'camera_params',
                'gaussian_splatting', 'rgb_video', 'depth_video'}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def save_obj(obj, name, meta):
    """Persist a Gradio FileData/ImageData result into OUT_DIR."""
    if obj is None:
        return None
    path = None
    if isinstance(obj, dict):
        path = obj.get('path')
        if not path:
            v = obj.get('video')
            # Gradio FileData sometimes stores `video` as a plain path string
            # instead of a nested dict -> guard both shapes.
            if isinstance(v, dict):
                path = v.get('path')
            elif isinstance(v, str) and os.path.exists(v):
                path = v
    elif isinstance(obj, str) and os.path.exists(obj):
        path = obj
    if path and os.path.exists(path):
        ext = os.path.splitext(path)[1] or '.bin'
        dest = os.path.join(OUT_DIR, name + ext)
        with open(path, 'rb') as src, open(dest, 'wb') as dst:
            dst.write(src.read())
        size = os.path.getsize(dest)
        log(f'saved {name} -> {os.path.basename(dest)} ({size} bytes)')
        return {'file': os.path.basename(dest), 'bytes': size}
    return {'raw': str(obj)[:160]}


def main():
    ap = argparse.ArgumentParser(description='Run genuine HunyuanWorld-Mirror inference.')
    ap.add_argument('--token', default=os.getenv('HF_TOKEN'), help='HuggingFace READ token')
    ap.add_argument('--space', default=DEFAULT_SPACE, help='Gradio Space URL')
    ap.add_argument('--image', default=DEFAULT_IMG, help='Input image path')
    ap.add_argument('--interval', type=float, default=1.0, help='Video sample interval')
    args = ap.parse_args()

    # If no token was passed, try the local HF cache (from `hf auth login`).
    if not args.token:
        try:
            from huggingface_hub import get_token as _get_token
            args.token = _get_token()
        except Exception:
            args.token = None

    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(args.image):
        log(f'ERROR: input image not found: {args.image}')
        sys.exit(1)

    meta = {
        'model': 'HunyuanWorld-Mirror',
        'provider': 'Tencent (official open-source world model)',
        'endpoint': args.space,
        'source_type': 'real_model',
        'authenticated': bool(args.token),
        'input_image': os.path.basename(args.image),
        'input_source': ('official Space example: examples/realistic/Bright_Room/image_0001.jpg'
                         if args.image == DEFAULT_IMG else 'user-supplied local file'),
        'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'steps': [],
    }

    try:
        log('connecting to Space' + (' (authenticated)' if args.token else ' (anonymous)') + '...')
        kwargs = {'verbose': False}
        if args.token:
            # gradio_client 2.x uses `token` (NOT `hf_token`) for Space auth.
            # This token is sent as HF authorization so ZeroGPU raises quota.
            kwargs['token'] = args.token
            # Also expose to huggingface_hub in case the backend checks cache.
            os.environ['HF_TOKEN'] = args.token
        c = Client(args.space, **kwargs)
        log('connected.')

        # Step 1: upload image -> server preprocesses into an input folder.
        log('step 1/2: uploading image + sampling views (cloud CPU)...')
        t0 = time.time()
        r1 = c.predict(api_name='/update_gallery_on_file_upload',
                       files=[handle_file(args.image)], interval=args.interval)
        output_path = r1[0]
        meta['steps'].append({
            'name': 'update_gallery_on_file_upload',
            'elapsed_sec': round(time.time() - t0, 1),
            'output_path': str(output_path),
            'terminal': str(r1[3])[:400] if len(r1) > 3 else '',
        })
        log(f'step 1 done in {meta["steps"][-1]["elapsed_sec"]}s: {output_path}')

        # Step 2: reconstruct -> real 3D artifacts (cloud GPU, needs quota).
        log('step 2/2: reconstructing 3D (Gaussian Splatting + depth + mesh)...')
        t0 = time.time()
        r2 = c.predict(api_name='/gradio_demo', target_dir=output_path,
                       frame_selector='All', show_camera=False,
                       filter_sky_bg=False, show_mesh=False, filter_ambiguous=False)
        meta['steps'].append({'name': 'gradio_demo', 'elapsed_sec': round(time.time() - t0, 1)})
        log(f'step 2 done in {meta["steps"][-1]["elapsed_sec"]}s.')

        meta['outputs'] = {}
        for i, item in enumerate(r2):
            label = OUTPUT_LABELS[i] if i < len(OUTPUT_LABELS) else f'item_{i}'
            if label in FILE_OUTPUTS:
                meta['outputs'][label] = save_obj(item, label, meta)
            elif label == 'terminal':
                meta['terminal_output'] = str(item)[:1200]

        meta['status'] = 'success'
        log('SUCCESS: genuine model outputs downloaded to real_run/output/')
    except Exception as e:
        msg = str(e)
        meta['status'] = 'error'
        meta['error'] = msg
        if 'ZeroGPU quota' in msg:
            meta['hint'] = ('Anonymous ZeroGPU quota exhausted. Register a free HuggingFace '
                            'account, create a READ token at https://huggingface.co/settings/tokens, '
                            'then rerun with: python real_run/run_inference.py --token hf_xxx')
        meta['traceback'] = traceback.format_exc()[-1200:]
        log('ERROR: ' + msg[:300])
    finally:
        meta['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(META_PATH, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        log(f'run metadata -> {META_PATH}')


if __name__ == '__main__':
    main()
