#!/usr/bin/env python3
"""
Person-detection alert monitor for the NVIDIA RT-VLM microservice.

Usage:
    pip install requests sseclient-py
    python monitor.py [--url http://localhost:8000] [--rtsp rtsp://...]
"""

import argparse
import json
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone

# Force line-buffered stdout so output is visible even when piped / redirected
sys.stdout.reconfigure(line_buffering=True)

import requests
import sseclient

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_RTSP_URL = os.getenv("RTSP_URL", "")
DEFAULT_RTSP_USER = os.getenv("RTSP_USER", "")
DEFAULT_RTSP_PASS = os.getenv("RTSP_PASS", "")

PERSON_PROMPT = """\
You are a security monitoring system. Analyze the video and determine if any \
person or people are visible in the scene and dropping a package, box, or parcel.
Respond with only a JSON object — no markdown fences, no extra text:
{"person_dropping_package_detected": true/false, "count": <integer>, "reason": "<one sentence describing what you see>"}\
"""

SYSTEM_PROMPT = (
    "You are a structured-output assistant. "
    "Always respond with a single raw JSON object exactly matching the schema requested. "
    "Do not include markdown, code fences, or any text outside the JSON object."
)

# Shorter chunk for faster alert latency on a live stream
CHUNK_DURATION = 30        # seconds per inference window
CHUNK_OVERLAP = 5          # seconds of overlap between windows

# ── State shared with signal handler ──────────────────────────────────────────

_stream_id: str | None = None
_base_url: str = DEFAULT_BASE_URL


def _deregister_stream() -> None:
    """Stop caption generation and deregister the stream (idempotent)."""
    global _stream_id
    sid = _stream_id
    if not sid:
        return
    print(f"\n[{_now()}] Stopping — cleaning up stream {sid} …")
    try:
        requests.delete(f"{_base_url}/generate_captions_alerts/{sid}", timeout=10)
    except Exception:
        pass
    try:
        requests.delete(f"{_base_url}/streams/delete/{sid}", timeout=10)
        print(f"[{_now()}] Stream removed.")
    except Exception:
        pass
    _stream_id = None


def _cleanup(signum=None, frame=None) -> None:
    """Signal handler: tear down stream then exit."""
    _deregister_stream()
    sys.exit(0)


signal.signal(signal.SIGINT, _cleanup)
signal.signal(signal.SIGTERM, _cleanup)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Steps ─────────────────────────────────────────────────────────────────────

def wait_for_ready(base_url: str, timeout: int = 600) -> None:
    """Poll /v1/ready until the service reports healthy."""
    print(f"[{_now()}] Waiting for RT-VLM service at {base_url} …")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/ready", timeout=5)
            if r.status_code == 200:
                print(f"[{_now()}] Service is ready.")
                return
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # Connection refused, DNS, TLS, etc.; read/connect timeouts (e.g. ReadTimeout)
            pass
        time.sleep(5)
    print(f"[{_now()}] ERROR: Service did not become ready within {timeout}s. Exiting.")
    sys.exit(1)


def get_model_id(base_url: str) -> str:
    """Return the ID of the first loaded model."""
    r = requests.get(f"{base_url}/models", timeout=10)
    r.raise_for_status()
    models = r.json().get("data", [])
    if not models:
        print(f"[{_now()}] ERROR: No models loaded. Check service logs.")
        sys.exit(1)
    model_id = models[0]["id"]
    print(f"[{_now()}] Loaded model: {model_id}")
    return model_id


def add_stream(base_url: str, rtsp_url: str, username: str = "", password: str = "") -> str:
    """Register the RTSP stream and return its stream ID.

    Credentials are passed as separate fields (not embedded in the URL) to
    comply with the AddLiveStream schema's strict pattern validation.
    The description must only contain: A-Z a-z 0-9 _ . - " ' space comma
    """
    global _stream_id
    stream_entry: dict = {
        "liveStreamUrl": rtsp_url,
        "description": "Security camera person detection",
        "sensor_name": "entrance-cam-01",
    }
    if username:
        stream_entry["username"] = username
    if password:
        stream_entry["password"] = password

    payload = {"streams": [stream_entry]}

    r = requests.post(f"{base_url}/streams/add", json=payload, timeout=30)
    if not r.ok:
        print(f"[{_now()}] ERROR {r.status_code} adding stream:\n{r.text}")
        sys.exit(1)
    data = r.json()
    if data.get("errors"):
        print(f"[{_now()}] ERROR adding stream: {data['errors']}")
        sys.exit(1)
    stream_id = data["results"][0]["id"]
    _stream_id = stream_id
    print(f"[{_now()}] Stream registered: {stream_id}")
    return stream_id


_FALLBACK: dict = {"person_dropping_package_detected": False, "count": 0, "reason": ""}


def _parse_chunk(content: str) -> dict:
    """Parse the VLM JSON response into structured fields.

    Strips optional markdown fences the model may emit before the JSON object,
    then falls back to safe defaults on any parse or type error.
    """
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return dict(_FALLBACK)
    try:
        data = json.loads(match.group())
        return {
            "person_dropping_package_detected": bool(
                data.get("person_dropping_package_detected", False)
            ),
            "count": int(data.get("count", 0)),
            "reason": str(data.get("reason", "")),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return dict(_FALLBACK)


def monitor_stream(base_url: str, stream_id: str, model_id: str) -> None:
    """Start SSE caption generation and print alerts when persons are detected."""
    print(f"[{_now()}] Starting person-detection monitor (chunk={CHUNK_DURATION}s) …")
    print("─" * 70)

    caption_payload = {
        "id": stream_id,
        "prompt": PERSON_PROMPT,
        "system_prompt": SYSTEM_PROMPT,
        "model": model_id,
        "stream": True,
        "chunk_duration": CHUNK_DURATION,
        "chunk_overlap_duration": CHUNK_OVERLAP,
        "enable_audio": False,
    }

    response = requests.post(
        f"{base_url}/generate_captions_alerts",
        json=caption_payload,
        stream=True,
        timeout=None,  # live stream — no overall timeout
    )
    response.raise_for_status()

    client = sseclient.SSEClient(response)
    for event in client.events():
        data = (event.data or "").strip()
        if not data or data == "[DONE]":
            if data == "[DONE]":
                print(f"[{_now()}] Stream ended ([DONE]).")
            break

        try:
            result = json.loads(data)
        except json.JSONDecodeError:
            continue

        for chunk in result.get("chunk_responses", []):
            content = chunk.get("content", "")
            t_start = chunk.get("start_time", "?")
            t_end = chunk.get("end_time", "?")
            parsed = _parse_chunk(content)

            if parsed["person_dropping_package_detected"]:
                print(
                    f"\n{'!'*70}\n"
                    f"  ALERT  [{_now()}]\n"
                    f"  Window : {t_start} -> {t_end}\n"
                    f"  Count  : {parsed['count']} person(s)\n"
                    f"  Reason : {parsed['reason']}\n"
                    f"{'!'*70}\n"
                )
            else:
                print(
                    f"[{_now()}] Clear  [{t_start} -> {t_end}]  "
                    f"{parsed['reason'] or 'No persons detected.'}"
                )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global _base_url

    parser = argparse.ArgumentParser(description="RT-VLM person-detection alert monitor")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/v1",
        help="Base URL of the RT-VLM service (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--rtsp",
        default=DEFAULT_RTSP_URL,
        help=f"RTSP stream URL without credentials (default: {DEFAULT_RTSP_URL})",
    )
    parser.add_argument(
        "--rtsp-user",
        default=DEFAULT_RTSP_USER,
        help=f"RTSP username (default: {DEFAULT_RTSP_USER})",
    )
    parser.add_argument(
        "--rtsp-pass",
        default=DEFAULT_RTSP_PASS,
        help="RTSP password",
    )
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Skip the readiness polling (service already running)",
    )
    args = parser.parse_args()

    _base_url = args.url

    if not args.skip_wait:
        wait_for_ready(_base_url)

    model_id = get_model_id(_base_url)
    stream_id = add_stream(_base_url, args.rtsp, args.rtsp_user, args.rtsp_pass)
    try:
        monitor_stream(_base_url, stream_id, model_id)
    finally:
        _deregister_stream()


if __name__ == "__main__":
    main()
