import launch

NETSTAT = """
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1012
  TCP    127.0.0.1:8501         0.0.0.0:0              LISTENING       23456
  TCP    127.0.0.1:8501         127.0.0.1:50123        ESTABLISHED     23456
  TCP    127.0.0.1:18501        0.0.0.0:0              LISTENING       999
"""


def test_finds_the_listening_process_on_windows():
    assert launch.parse_windows_netstat(NETSTAT, 8501) == 23456
    assert launch.parse_windows_netstat(NETSTAT, 9999) is None


def test_reuses_only_a_server_running_this_version(tmp_path, monkeypatch):
    info = tmp_path / "server.json"
    monkeypatch.setattr(launch, "SERVER_INFO", info)
    info.write_text('{"pid": 1, "version": "%s"}' % launch.current_version())
    assert launch.replace_stale_server() is True

    stopped = []
    info.write_text('{"pid": 1, "version": "old"}')
    monkeypatch.setattr(launch, "pid_on_port", lambda port=8501: 4242)
    monkeypatch.setattr(launch, "is_python", lambda pid: True)
    monkeypatch.setattr(launch, "stop", stopped.append)
    assert launch.replace_stale_server() is False and stopped == [4242]


def test_never_stops_a_program_that_is_not_python(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "SERVER_INFO", tmp_path / "missing.json")
    monkeypatch.setattr(launch, "pid_on_port", lambda port=8501: 4242)
    monkeypatch.setattr(launch, "is_python", lambda pid: False)
    monkeypatch.setattr(launch, "stop", lambda pid: (_ for _ in ()).throw(AssertionError("must not stop")))
    try:
        launch.replace_stale_server()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected SystemExit")
