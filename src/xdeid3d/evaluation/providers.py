"""
Sample providers for evaluation data.

This module provides utilities for loading and iterating over
evaluation samples from various sources (directories, videos, datasets).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np

from xdeid3d.evaluation.data import EvaluationSample, SampleType

__all__ = [
    "SampleProvider",
    "DirectorySampleProvider",
    "VideoSampleProvider",
    "PairedDirectorySampleProvider",
    "FrameSequenceProvider",
]


class SampleProvider(ABC):
    """
    Abstract base class for sample providers.

    Sample providers yield EvaluationSample objects from various
    data sources for evaluation.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[EvaluationSample]:
        """Iterate over samples."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples."""
        pass

    def __getitem__(self, index: int) -> EvaluationSample:
        """Get sample by index (default: iterate to index)."""
        for i, sample in enumerate(self):
            if i == index:
                return sample
        raise IndexError(f"Index {index} out of range")


class DirectorySampleProvider(SampleProvider):
    """
    Provides samples from a directory of image pairs.

    Expects either:
    - original/ and anonymized/ subdirectories with matching filenames
    - or original_* and anonymized_* prefixed files

    Args:
        directory: Root directory path
        extensions: Valid image extensions
        recursive: Search subdirectories
        load_images: Load images into memory (vs. paths)

    Example:
        >>> provider = DirectorySampleProvider("./data")
        >>> for sample in provider:
        ...     print(sample.sample_id)
    """

    def __init__(
        self,
        directory: Union[str, Path],
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp'),
        recursive: bool = False,
        load_images: bool = True,
    ):
        self.directory = Path(directory)
        self.extensions = extensions
        self.recursive = recursive
        self.load_images = load_images
        self._samples: Optional[List[EvaluationSample]] = None

    def _find_samples(self) -> List[EvaluationSample]:
        """Find and pair original/anonymized images."""
        samples = []

        # Check for original/ and anonymized/ subdirectories
        orig_dir = self.directory / "original"
        anon_dir = self.directory / "anonymized"

        if orig_dir.exists() and anon_dir.exists():
            samples = self._match_directories(orig_dir, anon_dir)
        else:
            # Look for prefix-based naming
            samples = self._match_prefixes()

        return sorted(samples, key=lambda s: s.sample_id)

    def _match_directories(
        self,
        orig_dir: Path,
        anon_dir: Path
    ) -> List[EvaluationSample]:
        """Match files from original/ and anonymized/ directories."""
        samples = []

        pattern = "**/*" if self.recursive else "*"
        for orig_path in orig_dir.glob(pattern):
            if orig_path.is_file() and orig_path.suffix.lower() in self.extensions:
                # Find corresponding anonymized file
                rel_path = orig_path.relative_to(orig_dir)
                anon_path = anon_dir / rel_path

                if anon_path.exists():
                    sample_id = str(rel_path.with_suffix(''))
                    samples.append(EvaluationSample(
                        sample_id=sample_id,
                        original=str(orig_path),
                        anonymized=str(anon_path),
                        sample_type=SampleType.IMAGE,
                    ))

        return samples

    def _match_prefixes(self) -> List[EvaluationSample]:
        """Match files with original_/anonymized_ prefixes."""
        samples = []
        orig_files: Dict[str, Path] = {}
        anon_files: Dict[str, Path] = {}

        pattern = "**/*" if self.recursive else "*"
        for path in self.directory.glob(pattern):
            if path.is_file() and path.suffix.lower() in self.extensions:
                name = path.stem.lower()
                if name.startswith('original_'):
                    key = name[9:]  # Remove prefix
                    orig_files[key] = path
                elif name.startswith('anonymized_'):
                    key = name[11:]
                    anon_files[key] = path

        # Match pairs
        for key in orig_files:
            if key in anon_files:
                samples.append(EvaluationSample(
                    sample_id=key,
                    original=str(orig_files[key]),
                    anonymized=str(anon_files[key]),
                    sample_type=SampleType.IMAGE,
                ))

        return samples

    def __iter__(self) -> Iterator[EvaluationSample]:
        if self._samples is None:
            self._samples = self._find_samples()

        for sample in self._samples:
            if self.load_images:
                sample.load_images()
            yield sample

    def __len__(self) -> int:
        if self._samples is None:
            self._samples = self._find_samples()
        return len(self._samples)


class PairedDirectorySampleProvider(SampleProvider):
    """
    Provides samples from two separate directories.

    Args:
        original_dir: Directory with original images
        anonymized_dir: Directory with anonymized images
        extensions: Valid image extensions
        match_by: How to match files ('name', 'index')

    Example:
        >>> provider = PairedDirectorySampleProvider(
        ...     original_dir="./originals",
        ...     anonymized_dir="./anonymized"
        ... )
    """

    def __init__(
        self,
        original_dir: Union[str, Path],
        anonymized_dir: Union[str, Path],
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp'),
        match_by: str = "name",
        load_images: bool = True,
    ):
        self.original_dir = Path(original_dir)
        self.anonymized_dir = Path(anonymized_dir)
        self.extensions = extensions
        self.match_by = match_by
        self.load_images = load_images
        self._samples: Optional[List[EvaluationSample]] = None

    def _find_samples(self) -> List[EvaluationSample]:
        """Find and match samples from both directories."""
        orig_files = sorted([
            f for f in self.original_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.extensions
        ])
        anon_files = sorted([
            f for f in self.anonymized_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.extensions
        ])

        samples = []

        if self.match_by == "name":
            # Match by filename (ignoring directory)
            anon_by_name = {f.stem: f for f in anon_files}
            for orig in orig_files:
                if orig.stem in anon_by_name:
                    samples.append(EvaluationSample(
                        sample_id=orig.stem,
                        original=str(orig),
                        anonymized=str(anon_by_name[orig.stem]),
                        sample_type=SampleType.IMAGE,
                    ))
        else:
            # Match by index (sorted order)
            for i, (orig, anon) in enumerate(zip(orig_files, anon_files)):
                samples.append(EvaluationSample(
                    sample_id=f"sample_{i:05d}",
                    original=str(orig),
                    anonymized=str(anon),
                    sample_type=SampleType.IMAGE,
                    frame_index=i,
                ))

        return samples

    def __iter__(self) -> Iterator[EvaluationSample]:
        if self._samples is None:
            self._samples = self._find_samples()

        for sample in self._samples:
            if self.load_images:
                sample.load_images()
            yield sample

    def __len__(self) -> int:
        if self._samples is None:
            self._samples = self._find_samples()
        return len(self._samples)


class VideoSampleProvider(SampleProvider):
    """
    Provides samples from video files.

    Args:
        original_video: Path to original video
        anonymized_video: Path to anonymized video
        frame_step: Extract every Nth frame
        max_frames: Maximum number of frames to extract

    Example:
        >>> provider = VideoSampleProvider(
        ...     original_video="original.mp4",
        ...     anonymized_video="anonymized.mp4",
        ...     frame_step=5
        ... )
    """

    def __init__(
        self,
        original_video: Union[str, Path],
        anonymized_video: Union[str, Path],
        frame_step: int = 1,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
    ):
        self.original_video = Path(original_video)
        self.anonymized_video = Path(anonymized_video)
        self.frame_step = frame_step
        self.max_frames = max_frames
        self.start_frame = start_frame
        self._num_frames: Optional[int] = None

    def _count_frames(self) -> int:
        """Count available frames."""
        import cv2
        cap = cv2.VideoCapture(str(self.original_video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        available = (total_frames - self.start_frame) // self.frame_step
        if self.max_frames:
            available = min(available, self.max_frames)

        return max(0, available)

    def __iter__(self) -> Iterator[EvaluationSample]:
        import cv2

        cap_orig = cv2.VideoCapture(str(self.original_video))
        cap_anon = cv2.VideoCapture(str(self.anonymized_video))

        fps = cap_orig.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        extracted = 0

        # Seek to start frame
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        cap_anon.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        while True:
            ret_orig, frame_orig = cap_orig.read()
            ret_anon, frame_anon = cap_anon.read()

            if not ret_orig or not ret_anon:
                break

            if frame_idx % self.frame_step == 0:
                timestamp_ms = (self.start_frame + frame_idx) / fps * 1000 if fps > 0 else 0

                sample = EvaluationSample(
                    sample_id=f"frame_{self.start_frame + frame_idx:06d}",
                    original=cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB),
                    anonymized=cv2.cvtColor(frame_anon, cv2.COLOR_BGR2RGB),
                    sample_type=SampleType.VIDEO,
                    frame_index=self.start_frame + frame_idx,
                    timestamp_ms=timestamp_ms,
                )
                yield sample

                extracted += 1
                if self.max_frames and extracted >= self.max_frames:
                    break

            frame_idx += 1

        cap_orig.release()
        cap_anon.release()

    def __len__(self) -> int:
        if self._num_frames is None:
            self._num_frames = self._count_frames()
        return self._num_frames


class FrameSequenceProvider(SampleProvider):
    """
    Provides samples from in-memory frame sequences.

    Args:
        original_frames: Sequence of original frames
        anonymized_frames: Sequence of anonymized frames
        poses: Optional sequence of (yaw, pitch) tuples
        metadata: Optional per-frame metadata

    Example:
        >>> provider = FrameSequenceProvider(
        ...     original_frames=orig_frames,
        ...     anonymized_frames=anon_frames,
        ...     poses=[(yaw, pitch) for yaw, pitch in camera_poses]
        ... )
    """

    def __init__(
        self,
        original_frames: Sequence[np.ndarray],
        anonymized_frames: Sequence[np.ndarray],
        poses: Optional[Sequence[Tuple[float, float]]] = None,
        depths: Optional[Sequence[np.ndarray]] = None,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ):
        if len(original_frames) != len(anonymized_frames):
            raise ValueError("Original and anonymized frame counts must match")

        self.original_frames = original_frames
        self.anonymized_frames = anonymized_frames
        self.poses = poses
        self.depths = depths
        self.metadata = metadata

    def __iter__(self) -> Iterator[EvaluationSample]:
        for i, (orig, anon) in enumerate(zip(self.original_frames, self.anonymized_frames)):
            yaw = pitch = None
            if self.poses and i < len(self.poses):
                yaw, pitch = self.poses[i]

            depth = None
            if self.depths and i < len(self.depths):
                depth = self.depths[i]

            meta = {}
            if self.metadata and i < len(self.metadata):
                meta = self.metadata[i]

            yield EvaluationSample(
                sample_id=f"frame_{i:06d}",
                original=orig,
                anonymized=anon,
                sample_type=SampleType.FRAME_SEQUENCE,
                frame_index=i,
                yaw=yaw,
                pitch=pitch,
                depth=depth,
                metadata=meta,
            )

    def __len__(self) -> int:
        return len(self.original_frames)

    def __getitem__(self, index: int) -> EvaluationSample:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range")

        yaw = pitch = None
        if self.poses and index < len(self.poses):
            yaw, pitch = self.poses[index]

        depth = None
        if self.depths and index < len(self.depths):
            depth = self.depths[index]

        meta = {}
        if self.metadata and index < len(self.metadata):
            meta = self.metadata[index]

        return EvaluationSample(
            sample_id=f"frame_{index:06d}",
            original=self.original_frames[index],
            anonymized=self.anonymized_frames[index],
            sample_type=SampleType.FRAME_SEQUENCE,
            frame_index=index,
            yaw=yaw,
            pitch=pitch,
            depth=depth,
            metadata=meta,
        )
