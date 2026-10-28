import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from core import ollama_admin
from core.ollama_admin import (
    OllamaAdminError,
    has_model,
    is_running,
    list_models,
    pull_model,
    recommended_model,
    total_ram_gb,
)

TAGS = {"models": [{"name": "qwen2.5:7b-instruct"}, {"name": "llama3:latest"}]}

PULL_OK = [
    {"status": "pulling manifest"},
    {"status": "pulling abc123", "digest": "sha256:abc", "total": 1000, "completed": 250},
    {"status": "pulling abc123", "digest": "sha256:abc", "total": 1000, "completed": 1000},
    {"status": "success"},
]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            self._send_json({"version": "0.5.0"})
        elif self.path == "/api/tags":
            self._send_json(TAGS)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(req)
        if req.get("model") == "missing:model":
            self._send_json({"error": "pull model manifest: file does not exist"}, 500)
            return
        lines = list(PULL_OK)
        if req.get("model") == "bad:stream":
            lines = [PULL_OK[0], {"error": "max retries exceeded"}]
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for line in lines:
            self.wfile.write(json.dumps(line).encode() + b"\n")
            self.wfile.flush()
        self.wfile.write(b"\n")


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.requests = []
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _closed_port_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


def test_is_running(server):
    _, url = server
    assert is_running(url) is True
    assert is_running(url + "/") is True


def test_is_running_false_on_closed_port():
    assert is_running(_closed_port_url(), timeout=0.5) is False


def test_list_models(server):
    _, url = server
    assert list_models(url) == ["qwen2.5:7b-instruct", "llama3:latest"]
    assert list_models(_closed_port_url()) == []


def test_has_model_tag_logic(server):
    _, url = server
    assert has_model(url, "qwen2.5:7b-instruct")
    assert not has_model(url, "qwen2.5")          # implies qwen2.5:latest, not installed
    assert not has_model(url, "qwen2.5:3b-instruct")
    assert has_model(url, "llama3")               # :latest implied
    assert has_model(url, "llama3:latest")
    assert not has_model(url, "llama3:8b")
    assert not has_model(_closed_port_url(), "llama3")


def test_pull_progress(server):
    srv, url = server
    events = list(pull_model(url, "qwen2.5:3b-instruct"))
    assert srv.requests == [{"model": "qwen2.5:3b-instruct", "stream": True}]
    assert events == [
        {"status": "pulling manifest", "completed": None, "total": None},
        {"status": "pulling abc123", "completed": 250, "total": 1000},
        {"status": "pulling abc123", "completed": 1000, "total": 1000},
        {"status": "success", "completed": None, "total": None},
    ]


def test_pull_error_line_raises(server):
    _, url = server
    gen = pull_model(url, "bad:stream")
    assert next(gen)["status"] == "pulling manifest"
    with pytest.raises(OllamaAdminError, match="max retries exceeded"):
        next(gen)


def test_pull_http_error_raises(server):
    _, url = server
    with pytest.raises(OllamaAdminError, match="file does not exist"):
        list(pull_model(url, "missing:model"))


def test_pull_unreachable_raises():
    with pytest.raises(OllamaAdminError, match="couldn't reach Ollama"):
        list(pull_model(_closed_port_url(), "x", timeout=0.5))


def test_recommended_model():
    assert recommended_model(4) == "qwen2.5:3b-instruct"
    assert recommended_model(7.9) == "qwen2.5:3b-instruct"
    assert recommended_model(8) == "qwen2.5:7b-instruct"
    assert recommended_model(32) == "qwen2.5:7b-instruct"
    assert recommended_model(None) == "qwen2.5:7b-instruct"


def test_total_ram_gb_real_and_failure(monkeypatch):
    value = total_ram_gb()
    assert value is None or value > 0

    def boom():
        raise OSError("nope")

    monkeypatch.setattr(ollama_admin.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ollama_admin, "_ram_linux", boom)
    monkeypatch.setattr(ollama_admin, "_ram_sysconf", boom)
    assert total_ram_gb() is None
