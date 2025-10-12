"""Preprocessing stages (audio + background SED tagging)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..logging_utils import StageGuard, _fmt_hms
from ..pipeline_checkpoint_system import ProcessingStage
from .base import PipelineState
from .utils import (
    compute_audio_sha16,
    compute_audio_sha16_from_file,
    compute_pp_signature,
    read_json_safe,
)

if TYPE_CHECKING:
    from ..orchestrator import AudioAnalysisPipelineV2

__all__ = ["run_preprocess", "run_background_sed"]


def run_preprocess(
    pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard
) -> None:
    if not hasattr(pipeline, "pre") or pipeline.pre is None:
        raise RuntimeError("Preprocessor component unavailable; initialization failed")

    # Compute preprocessing config signature early
    state.pp_sig = compute_pp_signature(pipeline.pp_conf)
    
    # Try to load cached preprocessed audio first
    state.audio_sha16 = compute_audio_sha16_from_file(state.input_audio_path)
    if state.audio_sha16:
        pipeline.checkpoints.seed_file_hash(state.input_audio_path, state.audio_sha16)
        
        cache_root = pipeline.cache_root
        cache_dir = cache_root / state.audio_sha16
        cache_dir.mkdir(parents=True, exist_ok=True)
        state.cache_dir = cache_dir
        
        preproc_cache_path = cache_dir / "preprocessed.npz"
        if preproc_cache_path.exists():
            try:
                cached = np.load(preproc_cache_path, allow_pickle=True)
                cached_sig = str(cached.get("pp_signature", ""))
                
                if cached_sig == state.pp_sig:
                    state.y = cached["audio"].astype(np.float32)
                    state.sr = int(cached["sample_rate"])
                    state.duration_s = float(cached["duration_s"])
                    
                    # Reconstruct health if present
                    if "health" in cached:
                        health_dict = cached["health"].item()
                        from ...pipeline.audio_preprocessing import AudioHealth
                        state.health = AudioHealth(**health_dict)
                    else:
                        state.health = None
                    
                    pipeline.corelog.info(
                        f"[preprocess] loaded cached preprocessed audio {_fmt_hms(state.duration_s)}"
                    )
                    pipeline.corelog.event(
                        "preprocess",
                        "cache_hit",
                        duration_s=state.duration_s,
                        cache_path=str(preproc_cache_path),
                    )
                    guard.done(duration_s=state.duration_s)
                    
                    # Still need to load diar/tx caches
                    _load_diar_tx_caches(pipeline, state, cache_dir)
                    return
                else:
                    pipeline.corelog.info("[preprocess] cache exists but config changed, re-preprocessing")
            except Exception as exc:
                pipeline.corelog.warn(f"[preprocess] failed to load cache: {exc}")
    
    # No cache or cache invalid - do actual preprocessing
    result = pipeline.pre.process_file(state.input_audio_path)
    state.y = np.asarray(result.audio, dtype=np.float32)
    state.sr = result.sample_rate
    state.health = result.health
    state.duration_s = float(result.duration_s)
    pipeline.corelog.info(f"[preprocess] file duration {_fmt_hms(state.duration_s)}")
    pipeline.corelog.event(
        "preprocess",
        "metrics",
        duration_s=state.duration_s,
        snr_db=float(getattr(result.health, "snr_db", 0.0)) if result.health else None,
    )
    guard.done(duration_s=state.duration_s)
    
    # Save preprocessed audio to cache
    if not state.audio_sha16:
        state.audio_sha16 = compute_audio_sha16(state.y)
        if state.audio_sha16:
            pipeline.checkpoints.seed_file_hash(state.input_audio_path, state.audio_sha16)

    
    pipeline.checkpoints.create_checkpoint(
        state.input_audio_path,
        ProcessingStage.AUDIO_PREPROCESSING,
        {"sr": state.sr},
        progress=5.0,
        file_hash=state.audio_sha16,
    )
    
    cache_root = pipeline.cache_root
    if not state.audio_sha16:
        cache_dir = cache_root / "nohash"
    else:
        cache_dir = cache_root / state.audio_sha16
    cache_dir.mkdir(parents=True, exist_ok=True)
    state.cache_dir = cache_dir
    
    # Save preprocessed audio to cache
    preproc_cache_path = cache_dir / "preprocessed.npz"
    try:
        health_dict = None
        if state.health:
            health_dict = {
                "snr_db": state.health.snr_db,
                "clipping_detected": state.health.clipping_detected,
                "silence_ratio": state.health.silence_ratio,
                "rms_db": state.health.rms_db,
                "est_lufs": state.health.est_lufs,
                "dynamic_range_db": state.health.dynamic_range_db,
                "floor_clipping_ratio": state.health.floor_clipping_ratio,
                "is_chunked": getattr(state.health, "is_chunked", False),
                "chunk_info": getattr(state.health, "chunk_info", None),
            }
        
        np.savez_compressed(
            preproc_cache_path,
            audio=state.y,
            sample_rate=state.sr,
            duration_s=state.duration_s,
            pp_signature=state.pp_sig,
            health=health_dict,
        )
        pipeline.corelog.info(f"[preprocess] saved cache to {preproc_cache_path}")
    except Exception as exc:
        pipeline.corelog.warn(f"[preprocess] failed to save cache: {exc}")
    
    # Load diar/tx caches
    _load_diar_tx_caches(pipeline, state, cache_dir)


def _load_diar_tx_caches(
    pipeline: AudioAnalysisPipelineV2, state: PipelineState, cache_dir: Path
) -> None:
    """Load diarization and transcription caches."""
    diar_path = cache_dir / "diar.json"
    tx_path = cache_dir / "tx.json"

    state.diar_cache = read_json_safe(diar_path)
    state.tx_cache = read_json_safe(tx_path)
    diar_cache_src = str(diar_path) if state.diar_cache else None
    tx_cache_src = str(tx_path) if state.tx_cache else None

    if not state.diar_cache or not state.tx_cache:
        for root in pipeline.cache_roots[1:]:
            alt_dir = Path(root) / state.audio_sha16
            alt_diar = read_json_safe(alt_dir / "diar.json")
            alt_tx = read_json_safe(alt_dir / "tx.json")
            if not state.diar_cache and alt_diar:
                state.diar_cache = alt_diar
                diar_cache_src = str(alt_dir / "diar.json")
            if not state.tx_cache and alt_tx:
                state.tx_cache = alt_tx
                tx_cache_src = str(alt_dir / "tx.json")
            if state.diar_cache and state.tx_cache:
                break

    def _cache_matches(obj: dict[str, object] | None) -> bool:
        return (
            bool(obj)
            and obj.get("version") == pipeline.cache_version
            and obj.get("audio_sha16") == state.audio_sha16
            and obj.get("pp_signature") == state.pp_sig
        )

    if _cache_matches(state.tx_cache):
        state.resume_tx = True
        state.resume_diar = bool(_cache_matches(state.diar_cache))
        if state.resume_diar:
            pipeline.corelog.info("[resume] using tx.json+diar.json caches; skipping diarize+ASR")
        else:
            pipeline.corelog.info(
                "[resume] using tx.json cache; skipping ASR and reconstructing turns from tx cache"
            )
        pipeline.corelog.event(
            "resume",
            "tx_cache_hit",
            audio_sha16=state.audio_sha16,
            src=tx_cache_src,
        )
    elif _cache_matches(state.diar_cache):
        state.resume_diar = True
        pipeline.corelog.info("[resume] using diar.json cache; skipping diarize")
        pipeline.corelog.event(
            "resume",
            "diar_cache_hit",
            audio_sha16=state.audio_sha16,
            src=diar_cache_src,
        )

    if pipeline.cfg.get("ignore_tx_cache"):
        state.diar_cache = None
        state.tx_cache = None
        state.resume_diar = False
        state.resume_tx = False


def run_background_sed(
    pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard
) -> None:
    empty_result = {"top": [], "dominant_label": None, "noise_score": 0.0}
    if not bool(pipeline.cfg.get("enable_sed", True)):
        disabled_result = dict(empty_result)
        disabled_result["enabled"] = False
        pipeline.corelog.info("[sed] background sound event detection disabled via configuration.")
        pipeline.stats.config_snapshot["background_sed"] = disabled_result
        state.sed_info = disabled_result
        guard.done()
        return

    sed_info = dict(empty_result)
    try:
        tagger = getattr(pipeline, "sed_tagger", None)
        if tagger is not None and state.y.size > 0 and state.sr:
            sed_info = dict(tagger.tag(state.y, state.sr) or empty_result)
            sed_info["enabled"] = True
            pipeline.corelog.event(
                "background_sed",
                "tags",
                dominant_label=sed_info.get("dominant_label"),
                noise_score=sed_info.get("noise_score"),
            )
        else:
            pipeline.corelog.warn(
                "[sed] tagger unavailable; emitting empty background tag summary."
            )
            sed_info["enabled"] = True
    except (
        ImportError,
        ModuleNotFoundError,
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:
        pipeline.corelog.warn(
            "[sed] tagging skipped: %s. Emitting empty background tag summary.",
            exc,
        )
        sed_info["enabled"] = True
    finally:
        tl_cfg = {
            "mode": str(pipeline.cfg.get("sed_mode", "auto")).lower(),
            "window_sec": float(pipeline.cfg.get("sed_window_sec", 1.0)),
            "hop_sec": float(pipeline.cfg.get("sed_hop_sec", 0.5)),
            "enter": float(pipeline.cfg.get("sed_enter", 0.5)),
            "exit": float(pipeline.cfg.get("sed_exit", 0.35)),
            "min_dur": pipeline.cfg.get("sed_min_dur", {}),
            "default_min_dur": float(pipeline.cfg.get("sed_default_min_dur", 0.30)),
            "merge_gap": float(pipeline.cfg.get("sed_merge_gap", 0.20)),
            "classmap_csv": pipeline.cfg.get("sed_classmap_csv"),
            "write_jsonl": bool(pipeline.cfg.get("sed_timeline_jsonl", False)),
            "median_k": int(pipeline.cfg.get("sed_median_k", 5)),
            "batch_size": int(pipeline.cfg.get("sed_batch_size", 256)),
            # Cap total windows processed in timeline to avoid long hangs on multi-hour inputs
            "max_windows": int(pipeline.cfg.get("sed_max_windows", 6000)),
        }

        run_timeline = False
        noise_score = float(sed_info.get("noise_score", 0.0) or 0.0)
        if tl_cfg["mode"] == "timeline":
            run_timeline = True
        elif tl_cfg["mode"] == "auto":
            run_timeline = noise_score >= 0.30

        if run_timeline and state.out_dir is not None:
            try:
                from ...affect.sed_timeline import run_sed_timeline

                model_paths = getattr(tagger, "model_paths", None)
                labels = getattr(tagger, "labels", None)
                file_id = pipeline.stats.file_id or Path(state.input_audio_path).name
                artifacts = run_sed_timeline(
                    state.y,
                    sr=state.sr,
                    cfg=tl_cfg,
                    out_dir=state.out_dir,
                    file_id=file_id,
                    model_paths=model_paths,
                    labels=labels,
                )
                if artifacts is not None:
                    sed_info["timeline_csv"] = str(artifacts.csv)
                    sed_info["timeline_jsonl"] = str(artifacts.jsonl) if artifacts.jsonl else None
                    sed_info["timeline_events"] = artifacts.events
                    sed_info["timeline_mode"] = tl_cfg["mode"]
                    if getattr(artifacts, "mode", None):
                        sed_info["timeline_inference_mode"] = artifacts.mode
            except Exception as exc:  # pragma: no cover - runtime dependent
                pipeline.corelog.warn(
                    "[sed.timeline] generation failed: %s. Falling back to global tags only.",
                    exc,
                )

        sed_info.setdefault("enabled", True)
        snapshot = dict(sed_info)
        events = snapshot.pop("timeline_events", None)
        if events is not None:
            snapshot["timeline_event_count"] = len(events)
        pipeline.stats.config_snapshot["background_sed"] = snapshot
        state.sed_info = sed_info
        guard.done()
