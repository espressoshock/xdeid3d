"""
I/O utilities for image and video handling.

Provides functions for loading and saving images and videos
with support for various formats and color space conversions.
"""

from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import numpy as np

__all__ = [
    "load_image",
    "save_image",
    "load_video_frames",
    "save_video",
    "ImageSequenceReader",
    "VideoReader",
]


def load_image(
    path: Union[str, Path],
    color_space: str = "RGB",
    resize: Optional[Tuple[int, int]] = None,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """Load image from file.

    Args:
        path: Path to image file
        color_space: Output color space (RGB, BGR, GRAY)
        resize: Optional (width, height) to resize to
        dtype: Output data type

    Returns:
        Image array (H, W, C) or (H, W) for grayscale

    Example:
        >>> img = load_image("photo.jpg")
        >>> img = load_image("photo.jpg", resize=(512, 512))
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Try OpenCV first
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        # Convert color space
        if color_space.upper() == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_space.upper() == "GRAY":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # BGR is default from OpenCV

        # Resize if requested
        if resize is not None:
            img = cv2.resize(img, resize, interpolation=cv2.INTER_LINEAR)

        return img.astype(dtype)

    except ImportError:
        pass

    # Fall back to PIL
    try:
        from PIL import Image

        img = Image.open(path)

        # Convert color space
        if color_space.upper() == "RGB":
            img = img.convert("RGB")
        elif color_space.upper() == "BGR":
            img = img.convert("RGB")
            arr = np.array(img)
            arr = arr[..., ::-1]  # RGB to BGR
            if resize is not None:
                from PIL import Image as PILImage
                img = PILImage.fromarray(arr)
                img = img.resize(resize, PILImage.LANCZOS)
                return np.array(img).astype(dtype)
            return arr.astype(dtype)
        elif color_space.upper() == "GRAY":
            img = img.convert("L")

        if resize is not None:
            img = img.resize(resize, Image.LANCZOS)

        return np.array(img).astype(dtype)

    except ImportError:
        raise ImportError("Either OpenCV or Pillow is required for image loading")


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    color_space: str = "RGB",
    quality: int = 95,
) -> None:
    """Save image to file.

    Args:
        image: Image array (H, W, C) or (H, W)
        path: Output file path
        color_space: Input color space (RGB, BGR, GRAY)
        quality: JPEG quality (1-100)

    Example:
        >>> save_image(img, "output.png")
        >>> save_image(img, "output.jpg", quality=90)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Try OpenCV first
    try:
        import cv2

        # Convert to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            if color_space.upper() == "RGB":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Set quality for JPEG
        params = []
        if path.suffix.lower() in [".jpg", ".jpeg"]:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif path.suffix.lower() == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - int(quality / 11)]

        cv2.imwrite(str(path), image, params)
        return

    except ImportError:
        pass

    # Fall back to PIL
    try:
        from PIL import Image

        if len(image.shape) == 3 and image.shape[2] == 3:
            if color_space.upper() == "BGR":
                image = image[..., ::-1]  # BGR to RGB
            img = Image.fromarray(image, mode="RGB")
        elif len(image.shape) == 2:
            img = Image.fromarray(image, mode="L")
        else:
            img = Image.fromarray(image)

        save_kwargs = {}
        if path.suffix.lower() in [".jpg", ".jpeg"]:
            save_kwargs["quality"] = quality

        img.save(path, **save_kwargs)

    except ImportError:
        raise ImportError("Either OpenCV or Pillow is required for image saving")


def load_video_frames(
    path: Union[str, Path],
    color_space: str = "RGB",
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    sample_rate: int = 1,
    resize: Optional[Tuple[int, int]] = None,
) -> Generator[np.ndarray, None, None]:
    """Load video frames as a generator.

    Args:
        path: Path to video file
        color_space: Output color space (RGB, BGR)
        start_frame: First frame to read
        max_frames: Maximum frames to read
        sample_rate: Read every Nth frame
        resize: Optional (width, height) to resize to

    Yields:
        Frame arrays (H, W, C)

    Example:
        >>> for frame in load_video_frames("video.mp4"):
        ...     process(frame)
        >>> frames = list(load_video_frames("video.mp4", max_frames=100))
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for video loading")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path}")

    try:
        frame_idx = 0
        yielded = 0

        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample rate check
            if (frame_idx - start_frame) % sample_rate != 0:
                frame_idx += 1
                continue

            # Convert color space
            if color_space.upper() == "RGB":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize if requested
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)

            yield frame
            yielded += 1

            if max_frames is not None and yielded >= max_frames:
                break

            frame_idx += 1

    finally:
        cap.release()


def save_video(
    frames: Union[List[np.ndarray], Generator],
    path: Union[str, Path],
    fps: float = 30.0,
    color_space: str = "RGB",
    codec: str = "mp4v",
) -> None:
    """Save frames as video.

    Args:
        frames: List or generator of frame arrays (H, W, C)
        path: Output video path
        fps: Frames per second
        color_space: Input color space (RGB, BGR)
        codec: FourCC codec code

    Example:
        >>> save_video(frames, "output.mp4", fps=30)
        >>> save_video(frame_generator(), "output.mp4")
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for video saving")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    writer = None

    try:
        for frame in frames:
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(
                    str(path),
                    fourcc,
                    fps,
                    (width, height),
                )

            # Convert color space
            if color_space.upper() == "RGB":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Ensure uint8
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

            writer.write(frame)

    finally:
        if writer is not None:
            writer.release()


class VideoReader:
    """Video file reader with random access.

    Args:
        path: Path to video file
        color_space: Output color space

    Example:
        >>> reader = VideoReader("video.mp4")
        >>> frame = reader[0]  # First frame
        >>> for frame in reader:
        ...     process(frame)
    """

    def __init__(
        self,
        path: Union[str, Path],
        color_space: str = "RGB",
    ):
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            raise ImportError("OpenCV is required for VideoReader")

        self.path = Path(path)
        self.color_space = color_space

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video: {self.path}")

        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        """Total number of frames."""
        return self._frame_count

    @property
    def fps(self) -> float:
        """Frames per second."""
        return self._fps

    @property
    def width(self) -> int:
        """Frame width."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height."""
        return self._height

    @property
    def duration(self) -> float:
        """Video duration in seconds."""
        return self._frame_count / self._fps if self._fps > 0 else 0

    def __len__(self) -> int:
        return self._frame_count

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get frame by index."""
        if idx < 0:
            idx = self._frame_count + idx

        if idx < 0 or idx >= self._frame_count:
            raise IndexError(f"Frame index {idx} out of range")

        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()

        if not ret:
            raise ValueError(f"Failed to read frame {idx}")

        if self.color_space.upper() == "RGB":
            frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

        return frame

    def __iter__(self):
        """Iterate over frames."""
        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            if self.color_space.upper() == "RGB":
                frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

            yield frame

    def close(self) -> None:
        """Release video capture."""
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ImageSequenceReader:
    """Image sequence reader for frame directories.

    Args:
        directory: Directory containing image frames
        pattern: Glob pattern for images
        color_space: Output color space

    Example:
        >>> reader = ImageSequenceReader("frames/", pattern="*.png")
        >>> for frame in reader:
        ...     process(frame)
    """

    def __init__(
        self,
        directory: Union[str, Path],
        pattern: str = "*.png",
        color_space: str = "RGB",
    ):
        self.directory = Path(directory)
        self.pattern = pattern
        self.color_space = color_space

        self._files = sorted(self.directory.glob(pattern))
        if not self._files:
            raise ValueError(f"No images found matching {pattern} in {directory}")

    @property
    def frame_count(self) -> int:
        """Number of frames."""
        return len(self._files)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get frame by index."""
        if idx < 0:
            idx = len(self._files) + idx

        if idx < 0 or idx >= len(self._files):
            raise IndexError(f"Frame index {idx} out of range")

        return load_image(self._files[idx], color_space=self.color_space)

    def __iter__(self):
        """Iterate over frames."""
        for path in self._files:
            yield load_image(path, color_space=self.color_space)

    def get_path(self, idx: int) -> Path:
        """Get file path for frame index."""
        return self._files[idx]
