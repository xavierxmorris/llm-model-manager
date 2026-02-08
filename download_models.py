"""
Direct HTTP download for HuggingFace models with resume support.
Bypasses the problematic XET/hf_transfer system entirely.
"""
import os
import sys
import time
import urllib.request
import ssl

MODELS_DIR = r"D:\LLM\models"

DOWNLOADS = [
    {
        "name": "Qwen3-Next-80B-A3B-Instruct-UD-Q2_K_XL.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF/resolve/main/Qwen3-Next-80B-A3B-Instruct-UD-Q2_K_XL.gguf",
        "size_gb": 30.1,
    },
    {
        "name": "gemma-3-27b-it-Q8_0.gguf",
        "url": "https://huggingface.co/ggml-org/gemma-3-27b-it-GGUF/resolve/main/gemma-3-27b-it-Q8_0.gguf",
        "size_gb": 28.7,
    },
    {
        "name": "mmproj-gemma-3-27b-it-f16.gguf",
        "url": "https://huggingface.co/ggml-org/gemma-3-27b-it-GGUF/resolve/main/mmproj-gemma-3-27b-it-f16.gguf",
        "size_gb": 0.858,
    },
]

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks


def format_size(b):
    if b >= 1073741824:
        return f"{b / 1073741824:.2f} GB"
    elif b >= 1048576:
        return f"{b / 1048576:.1f} MB"
    return f"{b / 1024:.0f} KB"


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}m {seconds % 60:.0f}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def download_file(url, dest_path, expected_gb):
    """Download a file with resume support and progress display."""
    partial_path = dest_path + ".partial"
    
    # Check if already complete
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        print(f"  SKIP: Already exists ({format_size(size)})")
        return True
    
    # Check for partial download
    resume_pos = 0
    if os.path.exists(partial_path):
        resume_pos = os.path.getsize(partial_path)
        print(f"  Resuming from {format_size(resume_pos)}...")
    
    # Build request with range header for resume
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    if resume_pos > 0:
        req.add_header("Range", f"bytes={resume_pos}-")
    
    # Create SSL context
    ctx = ssl.create_default_context()
    
    try:
        resp = urllib.request.urlopen(req, context=ctx, timeout=60)
    except Exception as e:
        print(f"  ERROR connecting: {e}")
        return False
    
    # Get total size
    content_length = resp.headers.get("Content-Length")
    if content_length:
        content_length = int(content_length)
    
    if resp.status == 206:  # Partial content (resume)
        total_size = resume_pos + (content_length or 0)
    elif resp.status == 200:
        total_size = content_length or int(expected_gb * 1073741824)
        if resume_pos > 0 and content_length and content_length == total_size:
            # Server doesn't support range, start over
            print(f"  Server doesn't support resume, restarting...")
            resume_pos = 0
    elif resp.status == 302 or resp.status == 301:
        print(f"  Redirect: {resp.status} — following...")
        return False
    else:
        print(f"  HTTP {resp.status}")
        return False
    
    print(f"  Total: {format_size(total_size)} | Downloading...")
    
    mode = "ab" if resume_pos > 0 else "wb"
    downloaded = resume_pos
    start_time = time.time()
    last_print = start_time
    last_bytes = downloaded
    stall_start = None
    
    try:
        with open(partial_path, mode) as f:
            while True:
                try:
                    chunk = resp.read(CHUNK_SIZE)
                except Exception as e:
                    print(f"\n  READ ERROR: {e}")
                    print(f"  Saved {format_size(downloaded)}. Re-run to resume.")
                    return False
                
                if not chunk:
                    break
                
                f.write(chunk)
                downloaded += len(chunk)
                
                now = time.time()
                
                # Stall detection (no data for 120s)
                if len(chunk) > 0:
                    stall_start = None
                else:
                    if stall_start is None:
                        stall_start = now
                    elif now - stall_start > 120:
                        print(f"\n  STALL DETECTED: No data for 120s. Saved {format_size(downloaded)}. Re-run to resume.")
                        return False
                
                # Progress every 5 seconds
                if now - last_print >= 5:
                    elapsed = now - start_time
                    interval_bytes = downloaded - last_bytes
                    interval_time = now - last_print
                    speed = interval_bytes / interval_time if interval_time > 0 else 0
                    
                    pct = (downloaded / total_size * 100) if total_size > 0 else 0
                    
                    remaining = total_size - downloaded
                    eta = remaining / speed if speed > 0 else 0
                    
                    avg_speed = (downloaded - resume_pos) / elapsed if elapsed > 0 else 0
                    
                    bar_width = 30
                    filled = int(bar_width * downloaded / total_size) if total_size > 0 else 0
                    bar = "#" * filled + "-" * (bar_width - filled)
                    
                    print(f"  [{bar}] {pct:5.1f}% | {format_size(downloaded)}/{format_size(total_size)} | {format_size(speed)}/s | ETA: {format_time(eta)} | Avg: {format_size(avg_speed)}/s", flush=True)
                    
                    last_print = now
                    last_bytes = downloaded
        
        # Download complete — rename partial to final
        elapsed = time.time() - start_time
        avg_speed = (downloaded - resume_pos) / elapsed if elapsed > 0 else 0
        print(f"\n  DONE: {format_size(downloaded)} in {format_time(elapsed)} (avg {format_size(avg_speed)}/s)")
        
        os.rename(partial_path, dest_path)
        return True
        
    except KeyboardInterrupt:
        print(f"\n  Interrupted. Saved {format_size(downloaded)}. Re-run to resume.")
        return False


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("=" * 70)
    print("  Direct Model Downloader (HTTP with Resume)")
    print("=" * 70)
    print(f"  Target: {MODELS_DIR}")
    print(f"  Models: {len(DOWNLOADS)}")
    total_gb = sum(d['size_gb'] for d in DOWNLOADS)
    print(f"  Total:  ~{total_gb:.1f} GB")
    print("=" * 70)
    
    results = []
    for i, dl in enumerate(DOWNLOADS, 1):
        print(f"\n[{i}/{len(DOWNLOADS)}] {dl['name']} (~{dl['size_gb']} GB)")
        dest = os.path.join(MODELS_DIR, dl['name'])
        ok = download_file(dl['url'], dest, dl['size_gb'])
        results.append((dl['name'], ok))
    
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    for name, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {name}")
    
    failed = [r for r in results if not r[1]]
    if failed:
        print(f"\n  {len(failed)} download(s) failed. Re-run to resume.")
        return 1
    else:
        print(f"\n  All downloads complete!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
