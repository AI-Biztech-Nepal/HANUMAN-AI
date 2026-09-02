"""
Asterisk ARI bridge — Phase 1 groundwork, NOT YET TESTED against a live
Asterisk instance (no VPS/trunk was available at the time this was written).

Bridges inbound PSTN calls (via Asterisk ARI + externalMedia) to the agent
core over the existing /ws/chat protocol, reusing the STT/TTS primitives in
media/voice_bridge.py. See docs/ASTERISK_SETUP.md for the dialplan snippet
and ari.conf/http.conf you need before this will do anything.

Call flow:
  dialplan -> Stasis(hanuman, ${EXTEN})
  -> StasisStart here -> answer, create a mixing bridge, create an
     externalMedia channel (Asterisk streams caller RTP audio to a UDP
     socket we open), add both channels to the bridge
  -> agent greeting synthesized + played via ARI channel `play`
  -> caller audio: UDP -> RTP depacketize -> energy-threshold VAD ->
     utterance WAV -> transcribe_wav() -> sent over /ws/chat -> reply text
  -> reply synthesized + played, loop until the agent ends the call
     (mirrors app/main.py's ws_chat closing on session.ended) or the
     caller hangs up (StasisEnd)

MUST run on the same host as Asterisk — EXTERNAL_MEDIA_HOST and
ASTERISK_SOUNDS_DIR (see app/config.py) both assume local filesystem and
network access to Asterisk.

Things that can only be confirmed against a real Asterisk box, and are
NOT verified by anything in this repo yet:
  - exact ARI REST parameter names/behavior for POST channels/externalMedia
    on your Asterisk version (this follows the documented ARI 18+ API)
  - the RTP framing Asterisk actually sends for format=slin16 (assumed:
    standard 12-byte header, CSRC list + extension handled, PCM16 mono
    16kHz payload)
  - whether format_wav is loaded so `sound:` playback of our generated
    WAVs works without extra codec config
  - end-of-call detection via the /ws/chat WebSocket closing, under real
    network conditions (retries/timeouts are NOT implemented)

No barge-in in this version — agent audio plays to completion before the
bridge listens for the caller again. Barge-in is future scaling work
(ARCHITECTURE.md §6), not part of the Phase 1 gate.

Run:  python media/asterisk_bridge.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
import sys
import tempfile
import uuid
import wave
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).parent))          # for voice_bridge
sys.path.insert(0, str(Path(__file__).parent.parent))   # for app.config / app.tenants

from voice_bridge import synthesize, transcribe_wav, voice_for_text  # noqa: E402
from app import config, tenants                          # noqa: E402

log = logging.getLogger("asterisk_bridge")

SAMPLE_RATE = 16000
AGENT_WS_URL = config.AGENT_WS_URL

# VAD tuning — energy-threshold silence detection, no external VAD dependency.
RMS_SPEECH_THRESHOLD = 500       # out of 32768 full-scale
SILENCE_TIMEOUT_MS = 700         # how long the caller must be quiet to end an utterance
MIN_UTTERANCE_MS = 300           # ignore blips shorter than this
MAX_UTTERANCE_MS = 6_000         # safety valve against a stuck-open mic


# ---------------------------------------------------------------- ARI REST

class AriClient:
    """Thin wrapper over the ARI REST API (httpx, basic auth)."""

    def __init__(self):
        self._http = httpx.AsyncClient(
            base_url=config.ARI_BASE_URL,
            auth=(config.ARI_USERNAME, config.ARI_PASSWORD),
            timeout=10.0,
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def answer_channel(self, channel_id: str) -> None:
        r = await self._http.post(f"/channels/{channel_id}/answer")
        r.raise_for_status()

    async def hangup_channel(self, channel_id: str) -> None:
        try:
            r = await self._http.delete(f"/channels/{channel_id}")
            r.raise_for_status()
        except httpx.HTTPStatusError:
            pass  # already gone — fine

    async def create_bridge(self) -> str:
        r = await self._http.post("/bridges", params={"type": "mixing"})
        r.raise_for_status()
        return r.json()["id"]

    async def destroy_bridge(self, bridge_id: str) -> None:
        try:
            r = await self._http.delete(f"/bridges/{bridge_id}")
            r.raise_for_status()
        except httpx.HTTPStatusError:
            pass

    async def add_channel_to_bridge(self, bridge_id: str, channel_id: str) -> None:
        # A freshly created channel (e.g. externalMedia) isn't always immediately
        # ready to be bridged — Asterisk can 422 briefly until it settles.
        last_exc: httpx.HTTPStatusError | None = None
        for attempt in range(5):
            r = await self._http.post(
                f"/bridges/{bridge_id}/addChannel", params={"channel": channel_id}
            )
            if r.status_code == 422:
                last_exc = httpx.HTTPStatusError(
                    f"422 on attempt {attempt}", request=r.request, response=r
                )
                await asyncio.sleep(0.1 * (attempt + 1))
                continue
            r.raise_for_status()
            return
        raise last_exc

    async def create_external_media_channel(self, external_host: str) -> str:
        r = await self._http.post(
            "/channels/externalMedia",
            params={
                "app": config.ARI_APP_NAME,
                "external_host": external_host,
                "format": "slin16",
                "encapsulation": "rtp",
                "transport": "udp",
                "connection_type": "client",
                "direction": "both",
            },
        )
        r.raise_for_status()
        return r.json()["id"]

    async def play_media(self, channel_id: str, media: str) -> str:
        r = await self._http.post(
            f"/channels/{channel_id}/play", params={"media": media}
        )
        r.raise_for_status()
        return r.json()["id"]  # playback id


# ---------------------------------------------------------------- UDP / RTP

class UdpAudioReceiver(asyncio.DatagramProtocol):
    """Receives RTP packets from Asterisk's externalMedia channel and pushes
    depacketized PCM16 payload onto an asyncio queue."""

    def __init__(self):
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()

    def datagram_received(self, data: bytes, addr) -> None:
        pcm = _strip_rtp_header(data)
        if pcm:
            self.queue.put_nowait(pcm)


def _strip_rtp_header(packet: bytes) -> bytes | None:
    """Standard 12-byte RTP header + optional CSRC list + optional extension."""
    if len(packet) < 12:
        return None
    first_byte = packet[0]
    cc = first_byte & 0x0F
    has_extension = bool(first_byte & 0x10)
    offset = 12 + (cc * 4)
    if has_extension:
        if len(packet) < offset + 4:
            return None
        ext_len_words = struct.unpack(">H", packet[offset + 2:offset + 4])[0]
        offset += 4 + (ext_len_words * 4)
    if offset >= len(packet):
        return None
    return packet[offset:]


async def open_udp_receiver(host: str) -> tuple[asyncio.DatagramTransport, UdpAudioReceiver, int]:
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.create_datagram_endpoint(
        UdpAudioReceiver, local_addr=(host, 0)
    )
    port = transport.get_extra_info("sockname")[1]
    return transport, protocol, port


# ---------------------------------------------------------------- VAD

class VadBuffer:
    """Energy-threshold speech/silence segmenter over a raw PCM16 stream."""

    def __init__(self):
        self._frames: list[bytes] = []
        self._speaking = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0

    def add_frame(self, pcm: bytes) -> bytes | None:
        """Feed one frame of PCM16 audio. Returns the completed utterance's
        PCM bytes once silence closes it out, else None."""
        if len(pcm) < 2:
            return None
        frame_ms = (len(pcm) / 2) / SAMPLE_RATE * 1000
        rms = _rms16(pcm)

        if rms >= RMS_SPEECH_THRESHOLD:
            self._speaking = True
            self._silence_ms = 0.0
            self._speech_ms += frame_ms
            self._frames.append(pcm)
        elif self._speaking:
            self._silence_ms += frame_ms
            self._speech_ms += frame_ms
            self._frames.append(pcm)
        else:
            return None

        timed_out = self._silence_ms >= SILENCE_TIMEOUT_MS
        overran = self._speech_ms >= MAX_UTTERANCE_MS
        if timed_out or overran:
            utterance = b"".join(self._frames)
            self._frames = []
            self._speaking = False
            self._silence_ms = 0.0
            self._speech_ms = 0.0
            if (len(utterance) / 2) / SAMPLE_RATE * 1000 < MIN_UTTERANCE_MS:
                return None
            return utterance
        return None


def _rms16(pcm: bytes) -> float:
    count = len(pcm) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack(f"<{count}h", pcm[: count * 2])
    return (sum(s * s for s in samples) / count) ** 0.5


def _write_wav(pcm: bytes, path: str) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


# ---------------------------------------------------------------- call handler

class CallHandler:
    def __init__(self, ari: AriClient, channel_id: str, dialed_number: str | None):
        self.ari = ari
        self.channel_id = channel_id
        self.dialed_number = dialed_number
        self.bridge_id: str | None = None
        self.external_channel_id: str | None = None
        self.udp_transport: asyncio.DatagramTransport | None = None
        self.udp_protocol: UdpAudioReceiver | None = None
        self.agent_ws = None
        self.hangup_event = asyncio.Event()
        self.sounds_dir = Path(config.ASTERISK_SOUNDS_DIR)
        self.sounds_subdir = self.sounds_dir.name or "custom"
        self.language = "ne"

    def on_caller_hangup(self) -> None:
        self.hangup_event.set()

    async def setup(self) -> str:
        """Answer the call, wire up the ARI bridge + external media, connect
        to the agent core, and return its greeting text (not yet spoken)."""
        self.udp_transport, self.udp_protocol, port = await open_udp_receiver(
            config.EXTERNAL_MEDIA_HOST
        )
        await self.ari.answer_channel(self.channel_id)
        self.bridge_id = await self.ari.create_bridge()
        await self.ari.add_channel_to_bridge(self.bridge_id, self.channel_id)

        external_host = f"{config.EXTERNAL_MEDIA_HOST}:{port}"
        self.external_channel_id = await self.ari.create_external_media_channel(external_host)
        await self.ari.add_channel_to_bridge(self.bridge_id, self.external_channel_id)

        tenant_id = tenants.tenant_for_number(self.dialed_number) if self.dialed_number else None
        cfg = tenants.get_or_default(tenant_id)
        # faster-whisper's language auto-detect ("auto") crashes on a chunk its
        # own VAD strips to nothing, which happens routinely on phone audio —
        # always pass a concrete language, defaulting to Nepali.
        self.language = cfg.language if cfg.language in ("ne", "en") else "ne"
        self.agent_ws = await websockets.connect(AGENT_WS_URL)
        await self.agent_ws.send(json.dumps({"tenant_id": cfg.tenant_id}))
        greeting = await self.agent_ws.recv()
        log.info("call %s: tenant=%s greeting=%r", self.channel_id, cfg.tenant_id, greeting)
        return greeting

    async def speak(self, text: str, pending_playbacks: dict[str, asyncio.Future]) -> None:
        if not text.strip():
            return
        stem = f"say_{uuid.uuid4().hex}"
        wav_path = self.sounds_dir / f"{stem}.wav"
        try:
            synthesize(text, str(wav_path), voice=voice_for_text(text))
        except Exception:
            log.exception("TTS failed for call %s", self.channel_id)
            return

        media = f"sound:{self.sounds_subdir}/{stem}"
        playback_id = await self.ari.play_media(self.channel_id, media)
        finished: asyncio.Future = asyncio.get_running_loop().create_future()
        pending_playbacks[playback_id] = finished
        try:
            await asyncio.wait_for(finished, timeout=30.0)
        except asyncio.TimeoutError:
            log.warning("playback %s never finished for call %s", playback_id, self.channel_id)
        finally:
            pending_playbacks.pop(playback_id, None)
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass

    async def listen_and_respond(self, pending_playbacks: dict[str, asyncio.Future]) -> None:
        """Main audio loop: drain caller RTP -> VAD -> STT -> agent -> TTS -> play."""
        vad = VadBuffer()
        while not self.hangup_event.is_set():
            try:
                pcm = await asyncio.wait_for(self.udp_protocol.queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            utterance = vad.add_frame(pcm)
            if utterance is None:
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                _write_wav(utterance, f.name)
                text = transcribe_wav(f.name, language=self.language)
            if not text.strip():
                continue

            log.info("call %s heard: %r", self.channel_id, text)
            try:
                await self.agent_ws.send(text)
                reply = await self.agent_ws.recv()
            except websockets.exceptions.ConnectionClosed:
                log.info("call %s: agent ended the conversation", self.channel_id)
                self.hangup_event.set()
                break

            await self.speak(reply, pending_playbacks)

    async def _cleanup(self) -> None:
        if self.agent_ws is not None:
            try:
                await self.agent_ws.close()
            except Exception:
                pass
        if self.udp_transport is not None:
            self.udp_transport.close()
        if self.external_channel_id:
            await self.ari.hangup_channel(self.external_channel_id)
        await self.ari.hangup_channel(self.channel_id)
        if self.bridge_id:
            await self.ari.destroy_bridge(self.bridge_id)


async def _run_call(ari: AriClient, channel_id: str, dialed_number: str | None,
                     pending_playbacks: dict[str, asyncio.Future],
                     active_calls: dict[str, CallHandler]) -> None:
    handler = CallHandler(ari, channel_id, dialed_number)
    active_calls[channel_id] = handler
    try:
        greeting = await handler.setup()
        await handler.speak(greeting, pending_playbacks)
        await handler.listen_and_respond(pending_playbacks)
    except Exception:
        log.exception("call %s crashed", channel_id)
    finally:
        await handler._cleanup()
        active_calls.pop(channel_id, None)


# ---------------------------------------------------------------- event loop

async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ari = AriClient()
    active_calls: dict[str, CallHandler] = {}
    active_tasks: dict[str, asyncio.Task] = {}
    pending_playbacks: dict[str, asyncio.Future] = {}

    ws_url = (
        config.ARI_BASE_URL.replace("http://", "ws://").replace("https://", "wss://")
        + f"/events?app={config.ARI_APP_NAME}&api_key={config.ARI_USERNAME}:{config.ARI_PASSWORD}"
    )
    log.info("connecting to ARI event stream at %s", ws_url)

    async with websockets.connect(ws_url) as ws:
        async for raw in ws:
            event = json.loads(raw)
            etype = event.get("type")

            if etype == "StasisStart":
                channel = event["channel"]
                channel_id = channel["id"]
                name = channel.get("name", "")
                if name.startswith("UnicastRTP/"):
                    continue  # our own external-media channel entering Stasis — ignore
                args = event.get("args") or []
                dialed = args[0] if args else channel.get("dialplan", {}).get("exten")
                log.info("StasisStart channel=%s dialed=%s", channel_id, dialed)
                active_tasks[channel_id] = asyncio.create_task(
                    _run_call(ari, channel_id, dialed, pending_playbacks, active_calls)
                )

            elif etype == "StasisEnd":
                channel_id = event.get("channel", {}).get("id")
                handler = active_calls.get(channel_id)
                if handler is not None:
                    handler.on_caller_hangup()
                active_tasks.pop(channel_id, None)

            elif etype == "PlaybackFinished":
                pid = event.get("playback", {}).get("id")
                fut = pending_playbacks.get(pid)
                if fut and not fut.done():
                    fut.set_result(None)

    await ari.close()


if __name__ == "__main__":
    asyncio.run(main())
