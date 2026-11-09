"""Configuration loading for ME3. Secrets live in the environment; settings in YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = ROOT / "config" / "settings.example.yaml"


class PathsConfig(BaseModel):
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    workspaces_dir: str = "workspaces"
    logs_dir: str = "logs"
    models_dir: str = "models"
    fd_benchmark_dir: str = "../fd-benchmark"


class AudioConfig(BaseModel):
    input_sample_rate: int = 16000
    input_channels: int = 1
    output_sample_rate: int = 22050
    output_channels: int = 1
    pcm_format: str = "pcm_f32le"
    input_device: int | None = None
    output_device: int | None = None
    input_device_hint: str = ""
    output_device_hint: str = ""
    capture_blocksize: int = 1024
    playback_blocksize: int = 1024
    capture_queue_max: int = 64


class SessionConfig(BaseModel):
    silence_normal_ms: int = 400
    silence_incomplete_ms: int = 900
    silence_clarification_ms: int = 1600
    # After TTS, ignore mic briefly so laptop speaker→mic echo is not transcribed.
    echo_hangover_ms: int = 450
    # When True, do not feed ASR while bot audio is playing (laptop no-AEC default).
    # Sustained loud speech still barge-in via VAD (see barge_in_*).
    gate_asr_while_speaking: bool = True
    # Laptop barge-in: VAD and/or energy spike over TTS echo baseline.
    barge_in_vad_threshold: float = 0.45
    barge_in_hold_ms: int = 120
    barge_in_hangover_ms: int = 200
    # Mic RMS must exceed echo EMA * ratio (and absolute floor) to count as barge.
    barge_in_energy_ratio: float = 2.2
    barge_in_energy_floor: float = 0.018
    ack_delay_ms: int = 700
    backchannel_enabled: bool = False
    backchannel_cooldown_ms: int = 10000
    overlap_yield_ms: int = 900
    discard_while_off: bool = True
    force_mock_llm: bool = False
    max_context_turns: int = 12
    timezone: str = "Asia/Manila"
    locale: str = "en_PH"


class ModelsConfig(BaseModel):
    asr: str = "sherpa-onnx-streaming-zipformer-en-2023-02-21-int8"
    tts: str = "en_US-lessac-medium"
    vad: str = "silero-vad"
    llm: str = "claude-haiku-4-5"
    asr_threads: int = 2
    tts_threads: int = 1
    vad_threads: int = 1


class SpeechConfig(BaseModel):
    asr_queue_max: int = 64
    tts_queue_max: int = 32
    asr_host: str = "127.0.0.1"
    asr_port: int = 8765
    tts_host: str = "127.0.0.1"
    tts_port: int = 8766
    vad_threshold: float = 0.5
    vad_window: int = 512


class UiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8780
    require_token: bool = True


class McpEndpointConfig(BaseModel):
    transport: str = "stdio"


class McpConfig(BaseModel):
    asr: McpEndpointConfig = Field(default_factory=McpEndpointConfig)
    tts: McpEndpointConfig = Field(default_factory=McpEndpointConfig)
    tasks: McpEndpointConfig = Field(default_factory=McpEndpointConfig)
    delegation: McpEndpointConfig = Field(default_factory=McpEndpointConfig)


class PendingConfig(BaseModel):
    """Flags that remain true until the corresponding setting is provisioned."""

    anthropic_api_key: bool = True
    tavily_api_key: bool = True
    github_pat: bool = True
    owner_public_profile: bool = True
    m1a_device_path: bool = True
    execution_permissions: bool = True
    opencode_server: bool = True


class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    speech: SpeechConfig = Field(default_factory=SpeechConfig)
    ui: UiConfig = Field(default_factory=UiConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    pending: PendingConfig = Field(default_factory=PendingConfig)


class EnvSecrets(BaseSettings):
    """Environment-backed secrets. Empty means pending / not provisioned."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    tavily_api_key: str = ""
    github_pat: str = ""
    github_owner: str = ""
    github_repo: str = ""
    github_branch: str = "main"
    github_site_root: str = "docs"
    github_pages_url: str = ""
    fdbench_url: str = ""
    me3_ui_token: str = ""
    # OpenCode local server (Phase 6) — not a cloud API key
    opencode_url: str = "http://127.0.0.1:4096"
    opencode_server_username: str = "opencode"
    opencode_server_password: str = ""
    opencode_force_mock: bool = False


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"settings file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"settings must be a mapping: {path}")
    return data


def settings_path() -> Path:
    override = os.environ.get("ME3_SETTINGS")
    if override:
        return Path(override)
    local = ROOT / "config" / "settings.yaml"
    if local.is_file():
        return local
    return DEFAULT_SETTINGS_PATH


def load_config(path: Path | None = None) -> AppConfig:
    cfg_path = path or settings_path()
    return AppConfig.model_validate(_load_yaml(cfg_path))


def load_secrets() -> EnvSecrets:
    return EnvSecrets()


def require_secret(name: str, value: str) -> str:
    if not value.strip():
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and provision credentials "
            "before enabling this feature (see pending flags in settings)."
        )
    return value


def resolve_path(relative: str, root: Path | None = None) -> Path:
    base = root or ROOT
    p = Path(relative)
    return p if p.is_absolute() else (base / p).resolve()


def models_dir(cfg: AppConfig | None = None) -> Path:
    cfg = cfg or load_config()
    return resolve_path(cfg.paths.models_dir)
