import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from core.jsonutil import extract_json_object
from core.llm_client import GeminiFreeTierClient, LLMError, OllamaClient, build_client


def test_extract_json_from_fence_and_prose():
    fenced = 'Sure!\n```json\n{"a": {"b": 1}}\n```\nDone.'
    assert extract_json_object(fenced) == {"a": {"b": 1}}
    assert extract_json_object('score: {"x": "}{", "y": 2} trailing') == {"x": "}{", "y": 2}
    with pytest.raises(ValueError):
        extract_json_object("no json here")


def test_build_client_switches_provider(monkeypatch):
    settings = {"llm": {"provider": "ollama", "ollama": {"model": "llama3.1:8b"}}}
    client = build_client(settings)
    assert isinstance(client, OllamaClient) and client.model == "llama3.1:8b"

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        build_client(settings, "gemini")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert isinstance(build_client(settings, "gemini"), GeminiFreeTierClient)

    monkeypatch.setenv("FJA_LLM_PROVIDER", "gemini")
    assert isinstance(build_client(settings), GeminiFreeTierClient)

    with pytest.raises(LLMError, match="Unknown"):
        build_client(settings, "openai")


@pytest.fixture
def stub_server():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, dict(self.headers), body))
            if self.path == "/api/chat":
                out = {"message": {"role": "assistant", "content": '{"ok": true}'}}
            else:
                out = {"choices": [{"message": {"content": '{"ok": true}'}}]}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", received
    server.shutdown()


def test_ollama_wire_format(stub_server):
    url, received = stub_server
    client = OllamaClient(base_url=url, model="qwen2.5:7b-instruct")
    assert client.chat("sys", "usr", json_mode=True, temperature=0.0) == '{"ok": true}'
    path, _, body = received[0]
    assert path == "/api/chat"
    assert body["format"] == "json" and body["stream"] is False
    assert body["options"]["temperature"] == 0.0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_gemini_wire_format(stub_server):
    url, received = stub_server
    client = GeminiFreeTierClient(api_key="k", base_url=url, min_interval=0)
    assert client.chat("sys", "usr", json_mode=True) == '{"ok": true}'
    path, headers, body = received[0]
    assert path == "/chat/completions"
    assert headers["Authorization"] == "Bearer k"
    assert body["response_format"] == {"type": "json_object"}


def test_unreachable_backend_raises_llm_error():
    client = OllamaClient(base_url="http://127.0.0.1:9", max_retries=0, timeout=2)
    with pytest.raises(LLMError, match="Cannot reach"):
        client.chat("s", "u")
