"""Safely download and normalize provider poster images into a shared cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

from app.utils import utils

NORMALIZATION_VERSION = "poster-jpeg-v1"
MAX_DOWNLOAD_BYTES = 3 * 1024 * 1024
MAX_DECODED_PIXELS = 12_000_000
MAX_NORMALIZED_EDGE = 1280
MAX_REDIRECTS = 2
CONNECT_TIMEOUT_SECONDS = 5
READ_TIMEOUT_SECONDS = 15
TOTAL_DEADLINE_SECONDS = 30
CHUNK_SIZE = 64 * 1024
NEUTRAL_ALPHA_BACKGROUND = (127, 127, 127, 255)
ALLOWED_HOSTS = {
    "pexels": frozenset({"images.pexels.com"}),
    "pixabay": frozenset({"cdn.pixabay.com"}),
}
_SOURCE_LOCKS = tuple(threading.Lock() for _ in range(128))


class PreviewError(Exception):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


class AggregateByteBudget:
    """A synchronized hard ceiling shared by concurrent response streams."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def consume(self, amount: int) -> bool:
        with self._lock:
            if amount < 0 or self.used + amount > self.limit:
                return False
            self.used += amount
            return True


@dataclass(frozen=True)
class PreparedPreview:
    image_sha256: str
    media_type: str
    width: int
    height: int
    byte_size: int
    cache_reference: str
    cache_hit: bool = False


def _cache_root() -> Path:
    return Path(utils.storage_dir("cache_candidate_previews", create=True))


def object_dir() -> Path:
    path = _cache_root() / "objects"
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_dir() -> Path:
    path = _cache_root() / "sources"
    path.mkdir(parents=True, exist_ok=True)
    return path


def object_name(digest: str) -> str:
    return f"{NORMALIZATION_VERSION}-{digest}.jpg"


def cache_reference(digest: str) -> str:
    return f"cache_candidate_previews/objects/{object_name(digest)}"


def _validated_url(provider: str, raw_url: str) -> str:
    try:
        parsed = urlsplit(raw_url)
        port = parsed.port
    except ValueError as exc:
        raise PreviewError("invalid_url", "invalid_url") from exc
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme.lower() != "https":
        raise PreviewError("invalid_url", "https_required")
    if parsed.username or parsed.password or (port is not None and port != 443):
        raise PreviewError("invalid_url", "embedded_credentials_or_port")
    if host not in ALLOWED_HOSTS.get(provider, frozenset()):
        raise PreviewError("invalid_url", "provider_host_not_allowed")
    if len(raw_url) > 4096:
        raise PreviewError("invalid_url", "invalid_url")
    return raw_url


def _source_key(provider: str, provider_video_id: str, preview_url: str) -> str:
    value = json.dumps(
        {
            "version": 1,
            "normalization_version": NORMALIZATION_VERSION,
            "provider": provider,
            "provider_video_id": provider_video_id,
            "preview_url": preview_url,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode()).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_cached(source_path: Path) -> PreparedPreview | None:
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if payload.get("normalization_version") != NORMALIZATION_VERSION:
            return None
        digest = payload["image_sha256"]
        path = object_dir() / object_name(digest)
        if path.is_symlink() or not path.is_file():
            return None
        data = path.read_bytes()
        if len(data) != payload["byte_size"]:
            return None
        if hashlib.sha256(data).hexdigest() != digest:
            return None
        return PreparedPreview(
            image_sha256=digest,
            media_type="image/jpeg",
            width=payload["width"],
            height=payload["height"],
            byte_size=len(data),
            cache_reference=cache_reference(digest),
            cache_hit=True,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def find_cached(
    provider: str, provider_video_id: str, preview_url: str
) -> PreparedPreview | None:
    url = _validated_url(provider, preview_url)
    key = _source_key(provider, provider_video_id, url)
    return _load_cached(source_dir() / f"source-v1-{key}.json")


def _download(
    provider: str,
    preview_url: str,
    byte_budget: AggregateByteBudget,
    session,
    monotonic,
) -> bytes:
    current = _validated_url(provider, preview_url)
    started = monotonic()
    for redirect_count in range(MAX_REDIRECTS + 1):
        if monotonic() - started > TOTAL_DEADLINE_SECONDS:
            raise PreviewError("download_failed", "total_deadline_exceeded")
        remaining = TOTAL_DEADLINE_SECONDS - (monotonic() - started)
        if remaining <= 0:
            raise PreviewError("download_failed", "total_deadline_exceeded")
        try:
            response = session.get(
                current,
                stream=True,
                allow_redirects=False,
                timeout=(
                    min(CONNECT_TIMEOUT_SECONDS, remaining),
                    min(READ_TIMEOUT_SECONDS, remaining),
                ),
                headers={
                    "User-Agent": "MoneyPrinterTurbo/scene-preview",
                    "Accept": "image/jpeg,image/png,image/webp",
                },
            )
        except requests.RequestException as exc:
            raise PreviewError("download_failed", "network_error") from exc
        with response:
            if response.status_code in {301, 302, 303, 307, 308}:
                if redirect_count >= MAX_REDIRECTS:
                    raise PreviewError("http_rejected", "redirect_limit_exceeded")
                location = response.headers.get("Location")
                if not location:
                    raise PreviewError("http_rejected", "redirect_not_allowed")
                current = _validated_url(provider, urljoin(current, location))
                continue
            if not 200 <= response.status_code < 300:
                raise PreviewError("http_rejected", "http_status_rejected")
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
            if content_type and content_type.lower() not in {
                "image/jpeg",
                "image/png",
                "image/webp",
            }:
                raise PreviewError("unsupported_format", "unsupported_content_type")
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > MAX_DOWNLOAD_BYTES:
                        raise PreviewError(
                            "size_limit_exceeded", "content_length_exceeded"
                        )
                except ValueError:
                    pass
            chunks = []
            size = 0
            try:
                iterator = response.iter_content(chunk_size=CHUNK_SIZE)
                for chunk in iterator:
                    if monotonic() - started > TOTAL_DEADLINE_SECONDS:
                        raise PreviewError("download_failed", "total_deadline_exceeded")
                    if not chunk:
                        continue
                    if not byte_budget.consume(len(chunk)):
                        raise PreviewError(
                            "budget_exhausted", "aggregate_byte_budget_exhausted"
                        )
                    if size + len(chunk) > MAX_DOWNLOAD_BYTES:
                        raise PreviewError(
                            "size_limit_exceeded", "stream_limit_exceeded"
                        )
                    chunks.append(chunk)
                    size += len(chunk)
            except requests.RequestException as exc:
                raise PreviewError("download_failed", "network_error") from exc
            return b"".join(chunks)
    raise PreviewError("http_rejected", "redirect_limit_exceeded")


def normalize_image(data: bytes) -> tuple[bytes, int, int]:
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise PreviewError("unsupported_format", "unsupported_decoded_format")
            if getattr(source, "n_frames", 1) != 1:
                raise PreviewError("unsupported_format", "animated_image")
            if source.width * source.height > MAX_DECODED_PIXELS:
                raise PreviewError(
                    "size_limit_exceeded", "decoded_pixel_limit_exceeded"
                )
            source.load()
            image = ImageOps.exif_transpose(source)
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, NEUTRAL_ALPHA_BACKGROUND)
                image = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                image = image.convert("RGB")
            if max(image.size) > MAX_NORMALIZED_EDGE:
                image.thumbnail(
                    (MAX_NORMALIZED_EDGE, MAX_NORMALIZED_EDGE), Image.Resampling.LANCZOS
                )
            output = BytesIO()
            image.save(
                output,
                format="JPEG",
                quality=85,
                optimize=False,
                progressive=False,
                subsampling=2,
            )
            return output.getvalue(), image.width, image.height
    except PreviewError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PreviewError("invalid_image", "decode_failed") from exc


def prepare_preview(
    provider: str,
    provider_video_id: str,
    preview_url: str,
    byte_budget: AggregateByteBudget,
    *,
    session=None,
    monotonic=time.monotonic,
) -> PreparedPreview:
    url = _validated_url(provider, preview_url)
    key = _source_key(provider, provider_video_id, url)
    source_path = source_dir() / f"source-v1-{key}.json"
    lock = _SOURCE_LOCKS[int(key[:8], 16) % len(_SOURCE_LOCKS)]
    with lock:
        cached = _load_cached(source_path)
        if cached:
            return cached
        wire_bytes = _download(
            provider, url, byte_budget, session or requests.Session(), monotonic
        )
        normalized, width, height = normalize_image(wire_bytes)
        digest = hashlib.sha256(normalized).hexdigest()
        object_path = object_dir() / object_name(digest)
        try:
            if not object_path.exists():
                _atomic_write(object_path, normalized)
            payload = json.dumps(
                {
                    "version": 1,
                    "normalization_version": NORMALIZATION_VERSION,
                    "image_sha256": digest,
                    "media_type": "image/jpeg",
                    "width": width,
                    "height": height,
                    "byte_size": len(normalized),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            _atomic_write(source_path, payload)
        except OSError as exc:
            raise PreviewError("storage_failed", "cache_write_failed") from exc
        return PreparedPreview(
            digest,
            "image/jpeg",
            width,
            height,
            len(normalized),
            cache_reference(digest),
        )
