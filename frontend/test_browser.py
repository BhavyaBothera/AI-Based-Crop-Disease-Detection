"""
Browser-level end-to-end check for the PHYTOSIGNAL input-safety flow.

WHAT IT VERIFIES
----------------
1. A photograph containing no plant tissue renders the rejection state, and the results
   card contains no disease name, candidate list, score or advisory controls.
2. "Upload another image" returns the page to the upload state.
3. A supported leaf photograph still produces the normal prediction, candidate list and
   agronomic reference.
4. Replacing a previously analysed leaf with a rejected image does not leave the old
   diagnosis on screen.
5. The same behaviour holds at a desktop (1440x900) and a mobile (390x844) viewport.

The harness starts its own uvicorn backend (with the real Keras model) and a static file
server, drives headless Chrome over the DevTools protocol, and writes screenshots to
docs/audit_reports/non_leaf_safety_screenshots/.

Run:
    .venv/Scripts/python.exe frontend/test_browser.py
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.tests.non_leaf_fixtures import create_portrait_bytes  # noqa: E402

SCREENSHOT_DIR = PROJECT_ROOT / "docs" / "audit_reports" / "non_leaf_safety_screenshots"
SAMPLE_LEAF = PROJECT_ROOT / "frontend" / "samples" / "sample_potato_early_blight.jpg"
# A real PlantVillage "background without leaves" photograph (snow-covered mountain).
# PlantVillage data is gitignored; the test is skipped if it is not present.
REAL_NON_LEAF = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "raw_dataset"
    / "Plant_leave_diseases_dataset_without_augmentation"
    / "Background_without_leaves"
    / "image (726).jpg"
)

VIEWPORTS = {
    "desktop": {"width": 1440, "height": 900, "mobile": False},
    "mobile": {"width": 390, "height": 844, "mobile": True},
}


class DevTools:
    """Minimal Chrome DevTools Protocol client."""

    def __init__(self, websocket_url: str):
        from websockets.sync.client import connect

        self._ws = connect(websocket_url, max_size=32 * 1024 * 1024)
        self._next_id = 0

    def send(self, method: str, params: dict = None):
        self._next_id += 1
        request_id = self._next_id
        self._ws.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self._ws.recv())
            if message.get("id") != request_id:
                continue  # event notification
            if "error" in message:
                raise RuntimeError(f"CDP {method} failed: {message['error']}")
            return message.get("result", {})

    def evaluate(self, expression: str):
        result = self.send(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        if result.get("exceptionDetails"):
            raise RuntimeError(f"JS evaluation failed: {result['exceptionDetails']}")
        return result.get("result", {}).get("value")

    def wait_for(self, expression: str, timeout: float = 60.0, interval: float = 0.25):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                last = self.evaluate(expression)
            except RuntimeError:
                # The execution context is briefly unavailable during navigation/reload.
                last = None
            if last:
                return last
            time.sleep(interval)
        raise AssertionError(f"Timed out waiting for: {expression} (last={last!r})")

    def close(self):
        try:
            self._ws.close()
        except Exception:
            pass


def wait_for_http(url: str, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code < 500:
                return
        except requests.RequestException:
            pass
        time.sleep(0.4)
    raise RuntimeError(f"Service at {url} did not become reachable within {timeout}s")


def start_services(backend_port: int, static_port: int, log_dir: Path):
    env = dict(os.environ)
    env["CORS_ORIGINS"] = (
        f"http://127.0.0.1:{static_port},http://localhost:{static_port},"
        "http://127.0.0.1:5500,http://localhost:5500"
    )
    backend_log = open(log_dir / "backend.log", "w", encoding="utf-8")
    backend = subprocess.Popen(
        [
            str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
            "-m",
            "uvicorn",
            "backend.app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(backend_port),
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=backend_log,
        stderr=subprocess.STDOUT,
    )
    static_log = open(log_dir / "static.log", "w", encoding="utf-8")
    static = subprocess.Popen(
        [
            str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
            "-m",
            "http.server",
            str(static_port),
            "--directory",
            str(PROJECT_ROOT / "frontend"),
        ],
        cwd=str(PROJECT_ROOT),
        stdout=static_log,
        stderr=subprocess.STDOUT,
    )
    wait_for_http(f"http://127.0.0.1:{static_port}/index.html", timeout=30)
    wait_for_http(f"http://127.0.0.1:{backend_port}/health", timeout=30)
    # Model load happens in the lifespan hook; /ready is 503 until it finishes.
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            payload = requests.get(f"http://127.0.0.1:{backend_port}/ready", timeout=3).json()
            if payload.get("model_loaded") and payload.get("knowledge_loaded"):
                return backend, static, backend_log, static_log
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("Backend did not report ready (model or knowledge base failed to load).")


def launch_chrome(chrome_path: Path, debug_port: int, profile_dir: Path) -> subprocess.Popen:
    chrome = subprocess.Popen(
        [
            str(chrome_path),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--hide-scrollbars",
            f"--remote-debugging-port={debug_port}",
            f"--user-data-dir={profile_dir}",
            "--window-size=1440,900",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            pages = requests.get(f"http://127.0.0.1:{debug_port}/json/list", timeout=2).json()
            for page in pages:
                if page.get("type") == "page" and page.get("webSocketDebuggerUrl"):
                    return chrome, page["webSocketDebuggerUrl"]
        except requests.RequestException:
            pass
        time.sleep(0.3)
    chrome.terminate()
    raise RuntimeError("Headless Chrome did not expose a DevTools page target.")


def set_viewport(devtools: DevTools, viewport: dict) -> None:
    devtools.send(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": viewport["width"],
            "height": viewport["height"],
            "deviceScaleFactor": 1,
            "mobile": viewport["mobile"],
        },
    )


def screenshot(devtools: DevTools, name: str, full_page: bool = True) -> Path:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    if full_page:
        # Reset scroll instantly so sticky headers are captured at the top of the image.
        devtools.evaluate(
            "document.documentElement.style.scrollBehavior='auto'; window.scrollTo(0, 0); true"
        )
        time.sleep(0.35)
    data = devtools.send(
        "Page.captureScreenshot",
        {"format": "png", "captureBeyondViewport": bool(full_page)},
    )["data"]
    path = SCREENSHOT_DIR / name
    path.write_bytes(base64.b64decode(data))
    return path


def screenshot_results(devtools: DevTools, name: str) -> Path:
    """Captures the visible results viewport (mobile evidence)."""
    devtools.evaluate(
        "document.documentElement.style.scrollBehavior='auto';"
        "document.querySelector('.results-card').scrollIntoView({block:'start'}); true"
    )
    time.sleep(0.35)
    return screenshot(devtools, name, full_page=False)


def upload_file(devtools: DevTools, path: Path) -> None:
    document = devtools.send("DOM.getDocument", {"depth": -1})
    node = devtools.send(
        "DOM.querySelector",
        {"nodeId": document["root"]["nodeId"], "selector": "#file-input"},
    )["nodeId"]
    if not node:
        raise AssertionError("Could not find #file-input")
    devtools.send("DOM.setFileInputFiles", {"files": [str(path)], "nodeId": node})
    devtools.wait_for(
        "!document.getElementById('preview-container').classList.contains('hidden')", timeout=20
    )


def analyze(devtools: DevTools, timeout: float = 90.0) -> dict:
    devtools.evaluate("document.getElementById('analyze-btn').click(); true")
    return devtools.wait_for(
        """(() => {
            if (!document.getElementById('loading-container').classList.contains('hidden')) return null;
            const rejected = !document.getElementById('results-rejected').classList.contains('hidden');
            const result = !document.getElementById('results-content').classList.contains('hidden');
            if (!rejected && !result) return null;
            const card = document.querySelector('.results-card');
            return {
                rejected,
                result,
                headline: document.getElementById('rejection-headline').textContent.trim(),
                detail: document.getElementById('rejection-detail').textContent.trim(),
                badge: document.getElementById('rejection-badge').textContent.trim(),
                cardText: card ? card.innerText : '',
                resultText: document.getElementById('results-content').innerText,
                candidates: document.querySelectorAll('#candidates-list .candidate-item').length,
                printVisible: !document.getElementById('print-advisory-btn').classList.contains('hidden'),
                copyVisible: !document.getElementById('copy-guidance-btn').classList.contains('hidden'),
                knowledgeVisible: !document.getElementById('knowledge-section').classList.contains('hidden'),
                rejectionVisible: rejected,
            };
        })()""",
        timeout=timeout,
    )


def assert_rejection(state: dict) -> None:
    assert state["rejected"], f"Expected rejection state, got {state}"
    assert not state["result"], "Results content must be hidden on rejection"
    assert "couldn't verify" in state["headline"].lower(), state["headline"]
    assert state["detail"], "Rejection detail should explain the problem"
    assert state["badge"], "Rejection badge should be set"
    assert not state["printVisible"], "Print advisory must be hidden on rejection"
    assert not state["copyVisible"], "Copy advisory must be hidden on rejection"
    assert not state["knowledgeVisible"], "Knowledge/advisory section must be hidden on rejection"
    assert state["candidates"] == 0, "No candidate cards may be rendered on rejection"
    forbidden = ("Late Blight", "Early Blight", "Bacterial Spot", "Septoria", "Leaf Mold", "Target Spot", "Healthy")
    for term in forbidden:
        assert term.lower() not in state["cardText"].lower(), f"Rejection UI leaked '{term}'"


def assert_prediction(state: dict) -> None:
    assert state["result"], f"Expected prediction state, got {state}"
    assert not state["rejected"], "Rejection state must be hidden for a supported leaf"
    assert state["candidates"] == 3, f"Expected 3 candidates, got {state['candidates']}"
    assert state["printVisible"], "Print control should be available for a valid prediction"
    assert state["copyVisible"], "Copy control should be available for a valid prediction"


def run_viewport(devtools: DevTools, label: str, viewport: dict, person_image: Path) -> None:
    set_viewport(devtools, viewport)
    devtools.send("Page.reload", {"ignoreCache": True})
    time.sleep(0.5)
    devtools.wait_for("document.readyState === 'complete'", timeout=30)
    readiness = "document.getElementById('connection-status-text').textContent === 'Model Ready'"
    try:
        devtools.wait_for(readiness, timeout=45)
    except AssertionError:
        # A single transient readiness-probe failure should not fail the suite: reload
        # once, then treat a second failure as a real regression.
        devtools.send("Page.reload", {"ignoreCache": True})
        time.sleep(1.0)
        devtools.wait_for("document.readyState === 'complete'", timeout=30)
        devtools.wait_for(readiness, timeout=45)

    # --- 1. Person photograph: must be rejected with no diagnosis -------------------
    upload_file(devtools, person_image)
    rejection = analyze(devtools)
    assert_rejection(rejection)
    if label == "mobile":
        # The sticky header must not cover the start of the rejection panel after the
        # automatic scroll-into-view.
        devtools.wait_for(
            "(() => { const h = document.querySelector('.app-header').getBoundingClientRect();"
            " const r = document.getElementById('results-rejected').getBoundingClientRect();"
            " return r.top >= h.bottom - 2; })()",
            timeout=10,
        )
    screenshot(devtools, f"{label}-person-rejected.png")
    if label == "mobile":
        screenshot_results(devtools, f"{label}-person-rejected-viewport.png")

    # --- 2. "Upload another image" restores the upload state ------------------------
    devtools.evaluate("document.getElementById('rejection-action-btn').click(); true")
    devtools.wait_for(
        "!document.getElementById('drop-zone').classList.contains('hidden') && "
        "document.getElementById('results-rejected').classList.contains('hidden')",
        timeout=10,
    )

    # --- 3. Supported leaf still produces the full prediction flow ------------------
    upload_file(devtools, SAMPLE_LEAF)
    prediction = analyze(devtools)
    assert_prediction(prediction)
    devtools.wait_for(
        "!document.getElementById('knowledge-details').classList.contains('hidden') || "
        "!document.getElementById('knowledge-error').classList.contains('hidden')",
        timeout=30,
    )
    screenshot(devtools, f"{label}-supported-leaf.png")
    if label == "mobile":
        screenshot_results(devtools, f"{label}-supported-leaf-viewport.png")

    # --- 4. Replacing the analysed leaf must clear the visible diagnosis -----------
    if REAL_NON_LEAF.is_file():
        upload_file(devtools, REAL_NON_LEAF)
        stale_check = analyze(devtools)
        assert_rejection(stale_check)
        screenshot(devtools, f"{label}-stale-result-cleared.png")
    else:
        print("  ! Real non-leaf photograph missing; skipping the stale-result probe.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-port", type=int, default=8010)
    parser.add_argument("--static-port", type=int, default=5500)
    parser.add_argument("--debug-port", type=int, default=9223)
    parser.add_argument(
        "--chrome",
        type=Path,
        default=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    )
    args = parser.parse_args()

    if not SAMPLE_LEAF.is_file():
        raise SystemExit(f"Missing sample leaf: {SAMPLE_LEAF}")
    if not args.chrome.is_file():
        raise SystemExit(f"Chrome not found at {args.chrome}")

    work_dir = PROJECT_ROOT / ".browser_test_tmp"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    person_image = work_dir / "person_probe.png"
    person_image.write_bytes(create_portrait_bytes())

    backend = static = chrome = None
    backend_log = static_log = None
    devtools = None
    failure = None
    try:
        print("Starting backend and static server...")
        backend, static, backend_log, static_log = start_services(
            args.backend_port, args.static_port, work_dir
        )
        print("Launching headless Chrome...")
        chrome, ws_url = launch_chrome(args.chrome, args.debug_port, work_dir / "profile")
        devtools = DevTools(ws_url)
        devtools.send("Page.enable")
        devtools.send("Runtime.enable")
        devtools.send("DOM.enable")
        devtools.send(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": f"window.API_BASE_URL='http://127.0.0.1:{args.backend_port}';"},
        )
        devtools.send(
            "Page.navigate", {"url": f"http://127.0.0.1:{args.static_port}/index.html"}
        )
        devtools.wait_for("document.readyState === 'complete'", timeout=30)

        for label, viewport in VIEWPORTS.items():
            print(f"Checking {label} viewport ({viewport['width']}x{viewport['height']})...")
            run_viewport(devtools, label, viewport, person_image)
            print(f"  {label}: OK")

        print(f"Saved screenshots to {SCREENSHOT_DIR}")
        return 0
    except Exception as exc:  # noqa: BLE001 - report any failure with service logs
        failure = exc
        print(f"BROWSER TEST FAILED: {exc}", file=sys.stderr)
        for log_path in (work_dir / "backend.log", work_dir / "static.log"):
            if log_path.is_file():
                print(f"--- {log_path.name} (tail) ---", file=sys.stderr)
                print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]), file=sys.stderr)
        return 1
    finally:
        if devtools:
            devtools.close()
        for proc in (chrome, static, backend):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for handle in (backend_log, static_log):
            if handle:
                handle.close()
        if failure is None and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
