"""Local file storage under data/ and build/. No database, no cloud."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.models import FitResult, Job
from core.settings import resolve_path


class Storage:
    def __init__(self, settings: dict[str, Any]):
        paths = settings.get("paths", {})
        self.data_dir = resolve_path(paths.get("data_dir", "data"))
        self.build_dir = resolve_path(paths.get("build_dir", "build"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.build_dir.mkdir(parents=True, exist_ok=True)

    @property
    def raw_jobs_path(self) -> Path:
        return self.data_dir / "raw_jobs.jsonl"

    @property
    def jobs_path(self) -> Path:
        return self.data_dir / "jobs.jsonl"          # sanitized + filtered

    @property
    def rankings_path(self) -> Path:
        return self.data_dir / "rankings.json"

    def job_build_dir(self, job_id: str) -> Path:
        d = self.build_dir / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)

    @staticmethod
    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def save_jobs(self, jobs: list[Job], path: Path | None = None) -> None:
        self.write_jsonl(path or self.jobs_path, [j.to_dict() for j in jobs])

    def load_jobs(self, path: Path | None = None) -> list[Job]:
        return [Job.from_dict(r) for r in self.read_jsonl(path or self.jobs_path)]

    def get_job(self, job_id: str) -> Job | None:
        return next((j for j in self.load_jobs() if j.id == job_id), None)

    def save_rankings(self, results: list[FitResult]) -> None:
        ordered = sorted(results, key=lambda r: r.fit_score, reverse=True)
        self.rankings_path.write_text(
            json.dumps([r.to_dict() for r in ordered], indent=2, ensure_ascii=False), encoding="utf-8")

    def load_rankings(self) -> list[FitResult]:
        if not self.rankings_path.exists():
            return []
        return [FitResult.from_dict(r) for r in json.loads(self.rankings_path.read_text(encoding="utf-8"))]
