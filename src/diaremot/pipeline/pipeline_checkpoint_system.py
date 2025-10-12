#!/usr/bin/env python3
"""
WhisperX Pipeline Checkpoint System
Handles incremental saves, resume functionality, and stage-based progress tracking
"""

import json
import logging
import pickle
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..utils.hash import hash_file


class ProcessingStage(Enum):
    """Pipeline processing stages"""

    AUDIO_PREPROCESSING = "audio_preprocessing"
    TRANSCRIPTION = "transcription"
    DIARIZATION = "diarization"
    EMOTION_ANALYSIS = "emotion_analysis"
    PARALINGUISTICS = "paralinguistics"
    CONVERSATION_ANALYSIS = "conversation_analysis"
    SUMMARY_GENERATION = "summary_generation"
    COMPLETE = "complete"


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint files"""

    stage: ProcessingStage
    timestamp: str
    file_hash: str
    audio_file: str
    progress_percent: float
    stage_data: dict[str, Any]
    total_segments: int = 0
    completed_segments: int = 0
    estimated_total_time: float = 0.0
    elapsed_time: float = 0.0


@dataclass
class ProgressState:
    """Current processing progress state"""

    current_stage: ProcessingStage
    stage_progress: float  # 0.0 to 1.0
    overall_progress: float  # 0.0 to 1.0
    segments_completed: int
    segments_total: int
    start_time: float
    last_checkpoint_time: float
    estimated_completion: float | None = None


class PipelineCheckpointManager:
    """Manages checkpoints and resume functionality for WhisperX pipeline"""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Stage weights for overall progress calculation
        self.stage_weights = {
            ProcessingStage.AUDIO_PREPROCESSING: 0.05,
            ProcessingStage.TRANSCRIPTION: 0.40,
            ProcessingStage.DIARIZATION: 0.20,
            ProcessingStage.EMOTION_ANALYSIS: 0.15,
            ProcessingStage.PARALINGUISTICS: 0.05,
            ProcessingStage.CONVERSATION_ANALYSIS: 0.10,
            ProcessingStage.SUMMARY_GENERATION: 0.05,
        }

        # Thread safety
        self._lock = threading.RLock()

        # Current state tracking
        self.current_progress: ProgressState | None = None
        self.auto_checkpoint_interval = 300  # 5 minutes

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Cache for file hashes during a single pipeline run
        self._hash_cache: dict[str, str] = {}

    def _cache_key(self, file_path: Path) -> str:
        file_path = Path(file_path)
        try:
            return str(file_path.resolve(strict=False))
        except Exception:
            return str(file_path)

    def _normalize_hash_value(self, value: str | None) -> str:
        if not value:
            return ""
        return value[:16]

    def _get_file_hash(self, file_path: Path, precomputed_hash: str | None = None) -> str:
        """Generate hash for file identification"""
        key = self._cache_key(file_path)

        with self._lock:
            if precomputed_hash:
                normalized = self._normalize_hash_value(precomputed_hash)
                self._hash_cache[key] = normalized
                return normalized

            if key in self._hash_cache:
                return self._hash_cache[key]

            if not file_path.exists():
                return ""

            digest = hash_file(file_path, open_func=open)
            normalized = self._normalize_hash_value(digest)
            self._hash_cache[key] = normalized
            return normalized

    def seed_file_hash(self, audio_file: str | Path, file_hash: str) -> None:
        """Seed the file-hash cache with a precomputed value."""
        self._get_file_hash(Path(audio_file), precomputed_hash=file_hash)

    def _get_checkpoint_path(
        self, audio_file: str, stage: ProcessingStage, file_hash: str | None = None
    ) -> Path:
        """Get checkpoint file path for audio file and stage"""
        audio_path = Path(audio_file)
        resolved_hash = self._get_file_hash(audio_path, precomputed_hash=file_hash)
        filename = f"{audio_path.stem}_{resolved_hash}_{stage.value}.checkpoint"
        return self.checkpoint_dir / filename

    def _get_metadata_path(
        self, audio_file: str, stage: ProcessingStage, file_hash: str | None = None
    ) -> Path:
        """Get metadata file path"""
        checkpoint_path = self._get_checkpoint_path(audio_file, stage, file_hash=file_hash)
        return checkpoint_path.with_suffix(".meta")

    def create_checkpoint(
        self,
        audio_file: str,
        stage: ProcessingStage,
        data: Any,
        progress: float = 0.0,
        stage_data: dict | None = None,
        file_hash: str | None = None,
    ) -> bool:
        """Create checkpoint for current stage"""
        with self._lock:
            try:
                checkpoint_path = self._get_checkpoint_path(audio_file, stage, file_hash=file_hash)
                metadata_path = self._get_metadata_path(audio_file, stage, file_hash=file_hash)
                audio_path = Path(audio_file)
                resolved_hash = self._get_file_hash(audio_path)

                # Save checkpoint data
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(data, f)

                # Create metadata
                metadata = CheckpointMetadata(
                    stage=stage,
                    timestamp=datetime.now().isoformat(),
                    file_hash=resolved_hash,
                    audio_file=audio_file,
                    progress_percent=progress,
                    stage_data=stage_data or {},
                    elapsed_time=(self._get_elapsed_time() if self.current_progress else 0.0),
                )

                # Save metadata (convert enum to string for JSON serialization)
                with open(metadata_path, "w") as f:
                    metadata_dict = asdict(metadata)
                    metadata_dict["stage"] = metadata.stage.value  # Convert enum to string
                    json.dump(metadata_dict, f, indent=2)

                self.logger.info(f"Checkpoint created: {stage.value} ({progress:.1f}%)")
                return True

            except Exception as e:
                self.logger.error(f"Failed to create checkpoint: {e}")
                return False

    def load_checkpoint(
        self,
        audio_file: str,
        stage: ProcessingStage,
        file_hash: str | None = None,
    ) -> tuple[Any, CheckpointMetadata]:
        """Load checkpoint data and metadata"""
        try:
            checkpoint_path = self._get_checkpoint_path(audio_file, stage, file_hash=file_hash)
            metadata_path = self._get_metadata_path(audio_file, stage, file_hash=file_hash)

            if not checkpoint_path.exists() or not metadata_path.exists():
                return None, None

            # Load metadata first
            with open(metadata_path) as f:
                metadata_dict = json.load(f)
                metadata_dict["stage"] = ProcessingStage(metadata_dict["stage"])
                metadata = CheckpointMetadata(**metadata_dict)

            # Verify file hasn't changed
            current_hash = self._get_file_hash(Path(audio_file))
            if current_hash != metadata.file_hash:
                self.logger.warning(f"File hash mismatch for {audio_file}, checkpoint may be stale")
                return None, None

            # Load checkpoint data
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)

            self.logger.info(f"Checkpoint loaded: {stage.value} ({metadata.progress_percent:.1f}%)")
            return data, metadata

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None, None

    def get_resume_point(self, audio_file: str) -> tuple[ProcessingStage, Any, CheckpointMetadata]:
        """Find the latest checkpoint to resume from"""
        audio_path = Path(audio_file)
        file_hash = self._get_file_hash(audio_path)
        hash_hint = file_hash or None
        latest_stage = None
        latest_data = None
        latest_metadata = None
        latest_timestamp = None

        # Check all stages for checkpoints
        for stage in ProcessingStage:
            if stage == ProcessingStage.COMPLETE:
                continue

            data, metadata = self.load_checkpoint(audio_file, stage, file_hash=hash_hint)
            if data is not None and metadata is not None:
                checkpoint_time = datetime.fromisoformat(metadata.timestamp)

                if latest_timestamp is None or checkpoint_time > latest_timestamp:
                    latest_stage = stage
                    latest_data = data
                    latest_metadata = metadata
                    latest_timestamp = checkpoint_time

        if latest_stage:
            self.logger.info(f"Resume point found: {latest_stage.value}")
            return latest_stage, latest_data, latest_metadata

        return None, None, None

    def start_processing(self, audio_file: str, total_segments: int = 0):
        """Initialize processing state"""
        with self._lock:
            self.current_progress = ProgressState(
                current_stage=ProcessingStage.AUDIO_PREPROCESSING,
                stage_progress=0.0,
                overall_progress=0.0,
                segments_completed=0,
                segments_total=total_segments,
                start_time=datetime.now().timestamp(),
                last_checkpoint_time=datetime.now().timestamp(),
            )

            self.logger.info(f"Started processing: {audio_file}")

    def update_progress(self, stage: ProcessingStage, progress: float, segments_completed: int = 0):
        """Update current progress"""
        with self._lock:
            if not self.current_progress:
                return

            self.current_progress.current_stage = stage
            self.current_progress.stage_progress = min(1.0, max(0.0, progress))
            self.current_progress.segments_completed = segments_completed

            # Calculate overall progress
            overall = 0.0

            for s, weight in self.stage_weights.items():
                if s == stage:
                    overall += weight * self.current_progress.stage_progress
                    break
                else:
                    overall += weight  # Previous stages are complete

            self.current_progress.overall_progress = min(1.0, overall)

            # Estimate completion time
            if self.current_progress.overall_progress > 0.05:
                elapsed = datetime.now().timestamp() - self.current_progress.start_time
                total_estimated = elapsed / self.current_progress.overall_progress
                remaining = total_estimated - elapsed
                self.current_progress.estimated_completion = remaining

    def should_checkpoint(self) -> bool:
        """Check if it's time for an automatic checkpoint"""
        if not self.current_progress:
            return False

        time_since_checkpoint = (
            datetime.now().timestamp() - self.current_progress.last_checkpoint_time
        )
        return time_since_checkpoint >= self.auto_checkpoint_interval

    def get_progress_summary(self) -> dict[str, Any]:
        """Get current progress summary"""
        if not self.current_progress:
            return {"status": "not_started"}

        elapsed = datetime.now().timestamp() - self.current_progress.start_time

        return {
            "status": "processing",
            "current_stage": self.current_progress.current_stage.value,
            "stage_progress": f"{self.current_progress.stage_progress * 100:.1f}%",
            "overall_progress": f"{self.current_progress.overall_progress * 100:.1f}%",
            "segments": f"{self.current_progress.segments_completed}/{self.current_progress.segments_total}",
            "elapsed_time": self._format_duration(elapsed),
            "estimated_remaining": (
                self._format_duration(self.current_progress.estimated_completion)
                if self.current_progress.estimated_completion
                else "calculating..."
            ),
            "last_checkpoint": self._format_duration(
                datetime.now().timestamp() - self.current_progress.last_checkpoint_time
            )
            + " ago",
        }

    def _format_duration(self, seconds: float) -> str:
        """Format duration as HH:MM:SS"""
        if seconds < 0:
            return "00:00:00"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _get_elapsed_time(self) -> float:
        """Get elapsed processing time"""
        if not self.current_progress:
            return 0.0
        return datetime.now().timestamp() - self.current_progress.start_time

    def cleanup_old_checkpoints(self, audio_file: str, keep_latest: int = 3):
        """Clean up old checkpoints for a file"""
        audio_path = Path(audio_file)
        file_hash = self._get_file_hash(audio_path)
        pattern = f"{audio_path.stem}_{file_hash}_*.checkpoint"

        checkpoint_files = list(self.checkpoint_dir.glob(pattern))
        if len(checkpoint_files) <= keep_latest:
            return

        # Sort by modification time
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old checkpoints
        removed_count = 0
        for checkpoint in checkpoint_files[keep_latest:]:
            try:
                checkpoint.unlink()
                # Remove corresponding metadata
                metadata_file = checkpoint.with_suffix(".meta")
                if metadata_file.exists():
                    metadata_file.unlink()
                removed_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")

        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} old checkpoints")

    def list_checkpoints(self, audio_file: str | None = None) -> list[dict[str, Any]]:
        """List available checkpoints"""
        checkpoints = []

        if audio_file:
            # List checkpoints for specific file
            audio_path = Path(audio_file)
            file_hash = self._get_file_hash(audio_path)
            pattern = f"{audio_path.stem}_{file_hash}_*.meta"
        else:
            # List all checkpoints
            pattern = "*.meta"

        for metadata_file in self.checkpoint_dir.glob(pattern):
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)

                checkpoints.append(
                    {
                        "audio_file": metadata.get("audio_file", "unknown"),
                        "stage": metadata.get("stage", "unknown"),
                        "progress": f"{metadata.get('progress_percent', 0):.1f}%",
                        "timestamp": metadata.get("timestamp", "unknown"),
                        "elapsed_time": self._format_duration(metadata.get("elapsed_time", 0)),
                        "file_size": (
                            metadata_file.with_suffix(".checkpoint").stat().st_size / 1024
                            if metadata_file.with_suffix(".checkpoint").exists()
                            else 0
                        ),
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to read checkpoint metadata {metadata_file}: {e}")

        return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)

    def mark_complete(self, audio_file: str):
        """Mark processing as complete and cleanup intermediate checkpoints"""
        with self._lock:
            try:
                # Create completion marker
                completion_data = {
                    "completed_at": datetime.now().isoformat(),
                    "total_time": self._get_elapsed_time(),
                    "stages_completed": list(self.stage_weights.keys()),
                }

                file_hash = self._get_file_hash(Path(audio_file)) or None

                self.create_checkpoint(
                    audio_file,
                    ProcessingStage.COMPLETE,
                    completion_data,
                    progress=100.0,
                    file_hash=file_hash,
                )

                # Reset progress state
                self.current_progress = None

                # Optional: cleanup intermediate checkpoints
                self.cleanup_old_checkpoints(audio_file, keep_latest=1)

                self.logger.info(f"Processing marked complete for {audio_file}")

            except Exception as e:
                self.logger.error(f"Failed to mark complete: {e}")

    def is_complete(self, audio_file: str) -> bool:
        """Check if processing is already complete for this file"""
        data, metadata = self.load_checkpoint(audio_file, ProcessingStage.COMPLETE)
        return data is not None

    def delete_checkpoints(self, audio_file: str) -> bool:
        """Delete all checkpoints for a specific audio file"""
        try:
            audio_path = Path(audio_file)
            file_hash = self._get_file_hash(audio_path)

            # Find all checkpoint and metadata files
            checkpoint_pattern = f"{audio_path.stem}_{file_hash}_*.checkpoint"
            metadata_pattern = f"{audio_path.stem}_{file_hash}_*.meta"

            removed_count = 0

            for pattern in [checkpoint_pattern, metadata_pattern]:
                for file_path in self.checkpoint_dir.glob(pattern):
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to delete {file_path}: {e}")

            self.logger.info(f"Deleted {removed_count} checkpoint files for {audio_file}")
            return removed_count > 0

        except Exception as e:
            self.logger.error(f"Failed to delete checkpoints: {e}")
            return False

    # Context manager for automatic checkpointing
    def processing_context(
        self, audio_file: str, stage: ProcessingStage, auto_checkpoint: bool = True
    ):
        """Context manager that automatically handles checkpointing"""
        return CheckpointContext(self, audio_file, stage, auto_checkpoint)


class CheckpointContext:
    """Context manager for automatic checkpoint handling"""

    def __init__(
        self,
        manager: PipelineCheckpointManager,
        audio_file: str,
        stage: ProcessingStage,
        auto_checkpoint: bool = True,
    ):
        self.manager = manager
        self.audio_file = audio_file
        self.stage = stage
        self.auto_checkpoint = auto_checkpoint
        self.data = None
        self.progress = 0.0

    def __enter__(self):
        # Try to load existing checkpoint
        self.data, metadata = self.manager.load_checkpoint(self.audio_file, self.stage)
        if self.data is not None:
            self.progress = metadata.progress_percent
            self.manager.logger.info(f"Resuming {self.stage.value} from {self.progress:.1f}%")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Save checkpoint on exit if no exception occurred
        if exc_type is None and self.auto_checkpoint:
            self.save_checkpoint()

    def update_progress(self, progress: float, data: Any = None):
        """Update progress within context"""
        self.progress = progress
        if data is not None:
            self.data = data

        # Update manager's progress tracking
        self.manager.update_progress(self.stage, progress / 100.0)

        # Auto-checkpoint if needed
        if self.auto_checkpoint and self.manager.should_checkpoint():
            self.save_checkpoint()

    def save_checkpoint(self):
        """Save current checkpoint"""
        if self.data is not None:
            success = self.manager.create_checkpoint(
                self.audio_file, self.stage, self.data, self.progress
            )
            if success:
                self.manager.current_progress.last_checkpoint_time = datetime.now().timestamp()


# Integration helper for existing WhisperX modules
class PipelineIntegration:
    """Helper class to integrate checkpoint system with existing pipeline"""

    def __init__(self, checkpoint_manager: PipelineCheckpointManager):
        self.checkpoint_manager = checkpoint_manager

    def wrap_transcription(self, transcription_func):
        """Wrap transcription function with checkpointing"""

        def wrapped_transcription(audio_file: str, **kwargs):
            with self.checkpoint_manager.processing_context(
                audio_file, ProcessingStage.TRANSCRIPTION
            ) as ctx:
                # Check if we can resume
                if ctx.data is not None:
                    # Resume from checkpoint
                    segments = ctx.data.get("segments", [])
                    completed_segments = len([s for s in segments if s.get("completed", False)])
                    start_segment = completed_segments
                else:
                    # Start fresh
                    segments = []
                    start_segment = 0
                    ctx.data = {"segments": segments, "metadata": {}}

                # Process segments incrementally
                total_segments = kwargs.get("total_segments", 100)  # Estimate

                for i in range(start_segment, total_segments):
                    # Process individual segment
                    try:
                        segment_result = transcription_func(audio_file, segment_index=i, **kwargs)
                        segment_result["completed"] = True

                        if i < len(segments):
                            segments[i] = segment_result
                        else:
                            segments.append(segment_result)

                        # Update progress
                        progress = ((i + 1) / total_segments) * 100
                        ctx.update_progress(progress, ctx.data)

                    except Exception as e:
                        # Log error but continue with next segment
                        logging.getLogger(__name__).error(f"Segment {i} failed: {e}")
                        continue

                return ctx.data

        return wrapped_transcription

    def wrap_diarization(self, diarization_func):
        """Wrap diarization function with checkpointing"""

        def wrapped_diarization(audio_file: str, transcription_data: dict, **kwargs):
            with self.checkpoint_manager.processing_context(
                audio_file, ProcessingStage.DIARIZATION
            ) as ctx:
                if ctx.data is not None:
                    # Resume from checkpoint
                    return ctx.data

                # Process diarization
                result = diarization_func(audio_file, transcription_data, **kwargs)
                ctx.update_progress(100.0, result)

                return result

        return wrapped_diarization

    def wrap_emotion_analysis(self, emotion_func):
        """Wrap emotion analysis with checkpointing"""

        def wrapped_emotion_analysis(audio_file: str, diarization_data: dict, **kwargs):
            with self.checkpoint_manager.processing_context(
                audio_file, ProcessingStage.EMOTION_ANALYSIS
            ) as ctx:
                if ctx.data is not None:
                    return ctx.data

                # Process emotions incrementally if possible
                segments = diarization_data.get("segments", [])
                results = {"segments": []}

                for i, segment in enumerate(segments):
                    emotion_result = emotion_func(segment, **kwargs)
                    results["segments"].append(emotion_result)

                    progress = ((i + 1) / len(segments)) * 100
                    ctx.update_progress(progress, results)

                return results

        return wrapped_emotion_analysis


def main():
    """CLI interface for checkpoint management"""
    import argparse

    parser = argparse.ArgumentParser(description="WhisperX Checkpoint Manager")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--action", choices=["list", "clean", "delete", "status"], required=True)
    parser.add_argument("--audio-file", help="Audio file path")
    parser.add_argument("--keep", type=int, default=3, help="Number of checkpoints to keep")

    args = parser.parse_args()

    manager = PipelineCheckpointManager(args.checkpoint_dir)

    if args.action == "list":
        checkpoints = manager.list_checkpoints(args.audio_file)
        if not checkpoints:
            print("No checkpoints found")
            return

        print(f"Found {len(checkpoints)} checkpoints:")
        print("-" * 80)
        print(f"{'Audio File':<25} {'Stage':<20} {'Progress':<10} {'Size (KB)':<10} {'Timestamp'}")
        print("-" * 80)

        for cp in checkpoints:
            audio_name = Path(cp["audio_file"]).name if cp["audio_file"] != "unknown" else "unknown"
            print(
                f"{audio_name:<25} {cp['stage']:<20} {cp['progress']:<10} {cp['file_size']:<10.1f} {cp['timestamp']}"
            )

    elif args.action == "clean":
        if not args.audio_file:
            print("Audio file required for clean action")
            return

        manager.cleanup_old_checkpoints(args.audio_file, args.keep)
        print(f"Cleaned old checkpoints for {args.audio_file}")

    elif args.action == "delete":
        if not args.audio_file:
            print("Audio file required for delete action")
            return

        success = manager.delete_checkpoints(args.audio_file)
        if success:
            print(f"Deleted all checkpoints for {args.audio_file}")
        else:
            print(f"Failed to delete checkpoints for {args.audio_file}")

    elif args.action == "status":
        summary = manager.get_progress_summary()
        print("=== Processing Status ===")
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
