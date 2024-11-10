import json
import logging
import math
import re
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import Any, Literal

from .utils import unique_hash


def chunk_by_chars(
    text: str, chunk_size: int = 2048, overlap: float = 0, threshold: int = 256
) -> list[str]:
    """
    Split a text into chunks of approximately equal size, with optional overlap.

    This function divides the input text into chunks based on the specified
    chunk size. It handles different scenarios based on the number of chunks
    required and provides options for overlap between chunks.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int, optional): The target size for each chunk. Defaults to 2048.
        overlap (float, optional): The fraction of overlap between chunks. Defaults to 0.
        threshold (int, optional): The minimum size for the last chunk. Defaults to 256.

    Returns:
        List[str]: A list of text chunks.

    Raises:
        ValueError: If an error occurs during the chunking process.

    Examples:
        >>> text = "This is a sample text for chunking."
        >>> chunks = chunk_by_chars(text, chunk_size=10, overlap=0.2)
        >>> print(chunks)
        ['This is a ', 'a sample ', 'le text fo', 'for chunki', 'king.']
    """
    try:
        n_chunks = math.ceil(len(text) / chunk_size)
        overlap_size = int(chunk_size * overlap / 2)

        if n_chunks == 1:
            return [text]
        elif n_chunks == 2:
            return _chunk_two_parts(text, chunk_size, overlap_size, threshold)
        else:
            return _chunk_multiple_parts(
                text, chunk_size, overlap_size, n_chunks, threshold
            )
    except Exception as e:
        raise ValueError(f"An error occurred while chunking the text: {e}")


def _chunk_two_parts(
    text: str, chunk_size: int, overlap_size: int, threshold: int
) -> list[str]:
    """Handle chunking for two parts."""
    first_chunk = text[: chunk_size + overlap_size]
    if len(text) - chunk_size > threshold:
        return [first_chunk, text[chunk_size - overlap_size :]]
    return [text]


def _chunk_multiple_parts(
    text: str,
    chunk_size: int,
    overlap_size: int,
    n_chunks: int,
    threshold: int,
) -> list[str]:
    """Handle chunking for more than two parts."""
    chunks = [text[: chunk_size + overlap_size]]

    for i in range(1, n_chunks - 1):
        start_idx = chunk_size * i - overlap_size
        end_idx = chunk_size * (i + 1) + overlap_size
        chunks.append(text[start_idx:end_idx])

    last_chunk_start = chunk_size * (n_chunks - 1) - overlap_size
    if len(text) - last_chunk_start > threshold:
        chunks.append(text[last_chunk_start:])
    else:
        chunks[-1] += text[chunk_size * (n_chunks - 1) + overlap_size :]

    return chunks


def chunk_by_tokens(
    tokens: list[str],
    chunk_size: int = 1024,
    overlap: float = 0,
    threshold: int = 128,
    return_tokens: bool = False,
) -> list[str | list[str]]:
    """
    Split a list of tokens into chunks of approximately equal size, with optional overlap.

    This function divides the input tokens into chunks based on the specified
    chunk size. It handles different scenarios based on the number of chunks
    required and provides options for overlap between chunks.

    Args:
        tokens (list[str]): The input list of tokens to be chunked.
        chunk_size (int, optional): The target size for each chunk. Defaults to 1024.
        overlap (float, optional): The fraction of overlap between chunks. Defaults to 0.
        threshold (int, optional): The minimum size for the last chunk. Defaults to 128.
        return_tokens (bool, optional): If True, return chunks as lists of tokens;
                                        if False, return as joined strings. Defaults to False.

    Returns:
        list[Union[str, list[str]]]: A list of chunked tokens, either as strings or token lists.

    Raises:
        ValueError: If an error occurs during the chunking process.

    Examples:
        >>> tokens = ["This", "is", "a", "sample", "text", "for", "chunking."]
        >>> chunks = chunk_by_tokens(tokens, chunk_size=3, overlap=0.2)
        >>> print(chunks)
        ['This is a', 'a sample text', 'text for chunking.']
    """
    try:
        n_chunks = math.ceil(len(tokens) / chunk_size)
        overlap_size = int(overlap * chunk_size / 2)
        residue = len(tokens) % chunk_size

        if n_chunks == 1:
            return _process_single_chunk(tokens, return_tokens)
        elif n_chunks == 2:
            return _chunk_two_parts(
                tokens,
                chunk_size,
                overlap_size,
                threshold,
                residue,
                return_tokens,
            )
        else:
            return _chunk_multiple_parts(
                tokens,
                chunk_size,
                overlap_size,
                n_chunks,
                threshold,
                residue,
                return_tokens,
            )
    except Exception as e:
        raise ValueError(f"An error occurred while chunking the tokens: {e}")


def _process_single_chunk(
    tokens: list[str], return_tokens: bool
) -> list[str | list[str]]:
    """Handle processing for a single chunk."""
    return [tokens] if return_tokens else [" ".join(tokens).strip()]


def _chunk_two_parts(
    tokens: list[str],
    chunk_size: int,
    overlap_size: int,
    threshold: int,
    residue: int,
    return_tokens: bool,
) -> list[str | list[str]]:
    """Handle chunking for two parts."""
    chunks = [tokens[: chunk_size + overlap_size]]
    if residue > threshold:
        chunks.append(tokens[chunk_size - overlap_size :])
    else:
        return _process_single_chunk(tokens, return_tokens)
    return _format_chunks(chunks, return_tokens)


def _chunk_multiple_parts(
    tokens: list[str],
    chunk_size: int,
    overlap_size: int,
    n_chunks: int,
    threshold: int,
    residue: int,
    return_tokens: bool,
) -> list[str | list[str]]:
    """Handle chunking for more than two parts."""
    chunks = [tokens[: chunk_size + overlap_size]]
    for i in range(1, n_chunks - 1):
        start_idx = chunk_size * i - overlap_size
        end_idx = chunk_size * (i + 1) + overlap_size
        chunks.append(tokens[start_idx:end_idx])

    last_chunk_start = chunk_size * (n_chunks - 1) - overlap_size
    if len(tokens) - last_chunk_start > threshold:
        chunks.append(tokens[last_chunk_start:])
    else:
        chunks[-1] += tokens[-residue:]

    return _format_chunks(chunks, return_tokens)


def _format_chunks(
    chunks: list[list[str]], return_tokens: bool
) -> list[str | list[str]]:
    """Format chunks based on the return_tokens flag."""
    return chunks if return_tokens else [" ".join(chunk).strip() for chunk in chunks]


def chunk_content(
    content: str,
    chunk_by: Literal["chars", "tokens"] = "chars",
    tokenizer: Callable[[str], list[str]] = str.split,
    chunk_size: int = 1024,
    overlap: float = 0,
    threshold: int = 256,
    metadata: dict[str, Any] = {},
    return_tokens: bool = False,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Split content into chunks and add metadata.

    This function takes a string content, splits it into chunks using the provided
    chunking function, and adds metadata to each chunk.

    Args:
        content (str): The content to be chunked.
        chunk_by(str): The method to use for chunking: "chars" or "tokens".
        tokenizer (Callable): The function to use for tokenization. defaults to str.split.
        chunk_size (int): The target size for each chunk.
        overlap (float): The fraction of overlap between chunks.
        threshold (int): The minimum size for the last chunk.
        metadata (Dict[str, Any]): Metadata to be included with each chunk.
        kwargs for tokenizer, if needed.


    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a chunk with metadata.
    """

    if chunk_by == "tokens":
        chunks = chunk_by_tokens(
            tokens=tokenizer(content, **kwargs),
            chunk_size=chunk_size,
            overlap=overlap,
            threshold=threshold,
            return_tokens=return_tokens,
        )
    else:
        chunks = chunk_by_chars(
            text=content,
            chunk_size=chunk_size,
            overlap=overlap,
            threshold=threshold,
        )

    return [
        {
            "chunk_content": chunk,
            "chunk_id": i + 1,
            "total_chunks": len(chunks),
            "chunk_size": len(chunk),
            **metadata,
        }
        for i, chunk in enumerate(chunks)
    ]


def clear_path(
    path: Path | str,
    /,
    recursive: bool = False,
    exclude: list[str] | None = None,
) -> None:
    """
    Clear all files and directories in the specified path.

    Args:
        path: The path to the directory to clear.
        recursive: If True, clears directories recursively.
        exclude: A list of string patterns to exclude from deletion.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        PermissionError: If there are insufficient permissions to delete
            files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"The specified directory {path} does not exist.")

    exclude = exclude or []
    exclude_pattern = re.compile("|".join(exclude)) if exclude else None

    for file_path in path.iterdir():
        if exclude_pattern and exclude_pattern.search(file_path.name):
            logging.info(f"Excluded from deletion: {file_path}")
            continue

        try:
            if file_path.is_dir():
                if recursive:
                    clear_path(file_path, recursive=True, exclude=exclude)
                    file_path.rmdir()
                else:
                    continue
            else:
                file_path.unlink()
            logging.info(f"Successfully deleted {file_path}")
        except PermissionError as e:
            logging.error(f"Permission denied when deleting {file_path}: {e}")
        except Exception as e:
            logging.error(f"Failed to delete {file_path}: {e}")


def copy_file(src: Path | str, dest: Path | str) -> None:
    """
    Copy a file from a source path to a destination path.

    Args:
        src: The source file path.
        dest: The destination file path.

    Raises:
        FileNotFoundError: If the source file does not exist or is not
            a file.
        PermissionError: If there are insufficient permissions to copy
            the file.
        OSError: If there's an OS-level error during the copy operation.
    """
    src_path, dest_path = Path(src), Path(dest)
    if not src_path.is_file():
        raise FileNotFoundError(f"{src_path} does not exist or is not a file.")

    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        copy2(src_path, dest_path)
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied when copying {src_path} to {dest_path}"
        ) from e
    except OSError as e:
        raise OSError(f"Failed to copy {src_path} to {dest_path}: {e}") from e


def create_path(
    directory: Path | str,
    filename: str,
    extension: str = None,
    timestamp: bool = False,
    dir_exist_ok: bool = True,
    file_exist_ok: bool = False,
    time_prefix: bool = False,
    timestamp_format: str | None = None,
    random_hash_digits: int = 0,
) -> Path:
    """
    Generate a new file path with optional timestamp and random hash.

    Args:
        directory: The directory where the file will be created.
        filename: The base name of the file to create.
        timestamp: If True, adds a timestamp to the filename.
        dir_exist_ok: If True, doesn't raise an error if the directory
            exists.
        file_exist_ok: If True, allows overwriting of existing files.
        time_prefix: If True, adds the timestamp as a prefix instead of
            a suffix.
        timestamp_format: Custom format for the timestamp.
        random_hash_digits: Number of digits for the random hash.

    Returns:
        The full path to the new or existing file.

    Raises:
        ValueError: If the filename contains illegal characters.
        FileExistsError: If the file exists and file_exist_ok is False.
    """
    if "/" in filename or "\\" in filename:
        raise ValueError("Filename cannot contain directory separators.")
    directory = Path(directory)

    name, ext = None, None
    if "." in filename:
        name, ext = filename.rsplit(".", 1)
    else:
        name = filename
        ext = extension.strip(".").strip() if extension else None

    if not ext:
        raise ValueError("No extension provided for filename.")

    ext = f".{ext}" if ext else ""

    if timestamp:
        timestamp_str = datetime.now().strftime(timestamp_format or "%Y%m%d%H%M%S")
        name = f"{timestamp_str}_{name}" if time_prefix else f"{name}_{timestamp_str}"

    if random_hash_digits > 0:
        random_hash = "-" + unique_hash(random_hash_digits)
        name = f"{name}{random_hash}"

    full_filename = f"{name}{ext}"
    full_path = directory / full_filename

    if full_path.exists():
        if file_exist_ok:
            return full_path
        raise FileExistsError(
            f"File {full_path} already exists and file_exist_ok is False."
        )
    full_path.parent.mkdir(parents=True, exist_ok=dir_exist_ok)
    return full_path


def _get_path_kwargs(
    persist_path: str | Path, postfix: str, **path_kwargs: Any
) -> dict[str, Any]:
    """
    Generate keyword arguments for path creation.

    Args:
        persist_path: The base path to use.
        postfix: The file extension to use.
        **path_kwargs: Additional keyword arguments to override defaults.

    Returns:
        A dictionary of keyword arguments for path creation.
    """
    persist_path = Path(persist_path)
    postfix = f".{postfix.strip('.')}"

    if persist_path.suffix != postfix:
        dirname = persist_path
        filename = f"new_file{postfix}"
    else:
        dirname, filename = persist_path.parent, persist_path.name

    return {
        "timestamp": path_kwargs.get("timestamp", False),
        "file_exist_ok": path_kwargs.get("file_exist_ok", True),
        "directory": path_kwargs.get("directory", dirname),
        "filename": path_kwargs.get("filename", filename),
    }


def dir_to_files(
    directory: str | Path,
    file_types: list[str] | None = None,
    max_workers: int | None = None,
    ignore_errors: bool = False,
    verbose: bool = False,
) -> list[Path]:
    """
    Recursively process a directory and return a list of file paths.

    This function walks through the given directory and its subdirectories,
    collecting file paths that match the specified file types (if any).

    Args:
        directory (Union[str, Path]): The directory to process.
        file_types (Optional[List[str]]): List of file extensions to include (e.g., ['.txt', '.pdf']).
                                          If None, include all file types.
        max_workers (Optional[int]): Maximum number of worker threads for concurrent processing.
                                     If None, uses the default ThreadPoolExecutor behavior.
        ignore_errors (bool): If True, log warnings for errors instead of raising exceptions.
        verbose (bool): If True, print verbose output.

    Returns:
        List[Path]: A list of Path objects representing the files found.

    Raises:
        ValueError: If the provided directory doesn't exist or isn't a directory.
    """
    directory_path = Path(directory)
    if not directory_path.is_dir():
        raise ValueError(f"The provided path is not a valid directory: {directory}")

    def process_file(file_path: Path) -> Path | None:
        try:
            if file_types is None or file_path.suffix in file_types:
                return file_path
        except Exception as e:
            if ignore_errors:
                if verbose:
                    logging.warning(f"Error processing {file_path}: {e}")
            else:
                raise ValueError(f"Error processing {file_path}: {e}") from e
        return None

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_file, f)
                for f in directory_path.rglob("*")
                if f.is_file()
            ]
            files = [
                future.result()
                for future in as_completed(futures)
                if future.result() is not None
            ]

        if verbose:
            logging.info(f"Processed {len(files)} files from {directory}")

        return files
    except Exception as e:
        raise ValueError(f"Error processing directory {directory}: {e}") from e


def file_to_chunks(
    file_path: str | Path,
    chunk_func: Callable[[str, int, float, int], list[str]],
    chunk_size: int = 1500,
    overlap: float = 0.1,
    threshold: int = 200,
    encoding: str = "utf-8",
    custom_metadata: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = False,
    timestamp: bool = True,
    random_hash_digits: int = 4,
) -> list[dict[str, Any]]:
    """
    Process a file and split its content into chunks.

    This function reads a file, splits its content into chunks using the provided
    chunking function, and optionally saves the chunks to separate files.

    Args:
        file_path (Union[str, Path]): Path to the file to be processed.
        chunk_func (Callable): Function to use for chunking the content.
        chunk_size (int): The target size for each chunk.
        overlap (float): The fraction of overlap between chunks.
        threshold (int): The minimum size for the last chunk.
        encoding (str): File encoding to use when reading the file.
        custom_metadata (Optional[Dict[str, Any]]): Additional metadata to include with each chunk.
        output_dir (Optional[Union[str, Path]]): Directory to save output chunks (if provided).
        verbose (bool): If True, print verbose output.
        timestamp (bool): If True, include timestamp in output filenames.
        random_hash_digits (int): Number of random hash digits to include in output filenames.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a chunk with metadata.

    Raises:
        ValueError: If there's an error processing the file.
    """
    try:
        file_path = Path(file_path)
        with open(file_path, encoding=encoding) as f:
            content = f.read()

        metadata = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            **(custom_metadata or {}),
        }

        chunks = chunk_content(
            content, chunk_func, chunk_size, overlap, threshold, metadata
        )

        if output_dir:
            save_chunks(chunks, output_dir, verbose, timestamp, random_hash_digits)

        return chunks
    except Exception as e:
        raise ValueError(f"Error processing file {file_path}: {e}") from e


def save_chunks(
    chunks: list[dict[str, Any]],
    output_dir: str | Path,
    verbose: bool,
    timestamp: bool,
    random_hash_digits: int,
) -> None:
    """Helper function to save chunks to files."""
    output_path = Path(output_dir)
    for i, chunk in enumerate(chunks):
        file_path = create_path(
            directory=output_path,
            filename=f"chunk_{i+1}",
            extension="json",
            timestamp=timestamp,
            random_hash_digits=random_hash_digits,
        )
        save_to_file(
            json.dumps(chunk, ensure_ascii=False, indent=2),
            directory=file_path.parent,
            filename=file_path.name,
            verbose=verbose,
        )


def get_file_size(path: Path | str) -> int:
    """
    Get the size of a file or total size of files in a directory.

    Args:
        path: The file or directory path.

    Returns:
        The size in bytes.

    Raises:
        FileNotFoundError: If the path does not exist.
        PermissionError: If there are insufficient permissions
            to access the path.
    """
    path = Path(path)
    try:
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        else:
            raise FileNotFoundError(f"{path} does not exist.")
    except PermissionError as e:
        raise PermissionError(f"Permission denied when accessing {path}") from e


def is_valid_path(
    path: str | Path,
    *,
    max_length: int | None = None,
    allow_relative: bool = True,
    allow_symlinks: bool = True,
    custom_reserved_names: list[str] | None = None,
    strict_mode: bool = False,
) -> bool:
    """
    Validates whether the given path is syntactically valid for the current operating system.

    Args:
        path (Union[str, Path]): The filesystem path to validate.
        max_length (Optional[int]): Maximum allowed path length. If None, uses OS default.
        allow_relative (bool): Whether to allow relative paths. Default is True.
        allow_symlinks (bool): Whether to allow symlinks. Default is True.
        custom_reserved_names (Optional[List[str]]): Additional reserved names to check.
        strict_mode (bool): If True, applies stricter validation rules. Default is False.

    Returns:
        bool: True if the path is valid, False otherwise.

    Raises:
        ValueError: If the path is invalid, with a detailed explanation.
    """
    if isinstance(path, Path):
        path_str = str(path)
    elif isinstance(path, str):
        path_str = path
    else:
        raise TypeError("Path must be a string or Path object.")

    if not path_str:
        raise ValueError("Path cannot be an empty string.")

    issues = []
    is_windows = sys.platform.startswith("win")

    # Common checks for both Windows and Unix-like systems
    if "\0" in path_str:
        issues.append("Path contains null character.")

    if not max_length:
        max_length = 260 if is_windows else 4096
    if len(path_str) > max_length:
        issues.append(f"Path exceeds the maximum length of {max_length} characters.")

    if is_windows:
        # Windows-specific validation
        invalid_chars = r'<>:"/\\|?*'
        if re.search(f"[{re.escape(invalid_chars)}]", path_str):
            issues.append(f"Path contains invalid characters: {invalid_chars}")

        reserved_names = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        }
        if custom_reserved_names:
            reserved_names.update(custom_reserved_names)

        path = Path(path_str)
        for part in path.parts:
            name = part.upper().rstrip(". ")
            if name in reserved_names:
                issues.append(f"Path contains a reserved name: '{part}'")

        if path_str.endswith(" ") or path_str.endswith("."):
            issues.append("Path cannot end with a space or a period on Windows.")

        if strict_mode:
            if not path_str.startswith("\\\\?\\") and len(path_str) > 260:
                issues.append("Path exceeds 260 characters without long path prefix.")

    else:
        # Unix-like systems validation
        if strict_mode:
            if re.search(r"//+", path_str):
                issues.append("Path contains consecutive slashes.")

        if not allow_relative and not path_str.startswith("/"):
            issues.append("Relative paths are not allowed.")

    # Common additional checks
    if not allow_symlinks and Path(path_str).is_symlink():
        issues.append("Symlinks are not allowed.")

    if strict_mode:
        if re.search(r"\s", path_str):
            issues.append("Path contains whitespace characters.")

    if issues:
        raise ValueError("Invalid path: " + "; ".join(issues))

    return True


def list_files(dir_path: Path | str, extension: str | None = None) -> list[Path]:
    """
    List all files in a specified directory with an optional extension
    filter, including files in subdirectories.

    Args:
        dir_path: The directory path where files are listed.
        extension: Filter files by extension.

    Returns:
        A list of Path objects representing files in the directory.

    Raises:
        NotADirectoryError: If the provided dir_path is not a directory.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{dir_path} is not a directory.")

    pattern = f"*.{extension}" if extension else "*"
    return [f for f in dir_path.rglob(pattern) if f.is_file()]


def read_file(path: Path | str, /) -> str:
    """
    Read the contents of a file.

    Args:
        path: The path to the file to read.

    Returns:
        str: The contents of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If there are insufficient permissions to read
            the file.
    """
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as e:
        logging.error(f"File not found: {path}: {e}")
        raise
    except PermissionError as e:
        logging.error(f"Permission denied when reading file: {path}: {e}")
        raise


def save_to_file(
    text: str,
    directory: Path | str,
    filename: str,
    extension: str = None,
    timestamp: bool = False,
    dir_exist_ok: bool = True,
    file_exist_ok: bool = False,
    time_prefix: bool = False,
    timestamp_format: str | None = None,
    random_hash_digits: int = 0,
    verbose: bool = True,
) -> Path:
    """
    Save text to a file within a specified directory, optionally adding a
    timestamp, hash, and verbose logging.

    Args:
        text: The text to save.
        directory: The directory path to save the file.
        filename: The filename for the saved text.
        timestamp: If True, append a timestamp to the filename.
        dir_exist_ok: If True, creates the directory if it does not exist.
        time_prefix: If True, prepend the timestamp instead of appending.
        timestamp_format: A custom format for the timestamp.
        random_hash_digits: Number of random hash digits to append
            to filename.
        verbose: If True, logs the file path after saving.

    Returns:
        Path: The path to the saved file.

    Raises:
        OSError: If there's an error creating the directory or
            writing the file.
    """
    try:
        file_path = create_path(
            directory=directory,
            filename=filename,
            extension=extension,
            timestamp=timestamp,
            dir_exist_ok=dir_exist_ok,
            file_exist_ok=file_exist_ok,
            time_prefix=time_prefix,
            timestamp_format=timestamp_format,
            random_hash_digits=random_hash_digits,
        )
        with file_path.open("w", encoding="utf-8") as file:
            file.write(text)
        if verbose:
            logging.info(f"Text saved to: {file_path}")
        return file_path

    except OSError as e:
        logging.error(f"Failed to save file {filename}: {e}")
        raise


def split_path(path: Path | str) -> tuple[Path, str]:
    """
    Split a path into its directory and filename components.

    Args:
        path: The path to split.

    Returns:
        A tuple containing the directory and filename.
    """
    path = Path(path)
    return path.parent, path.name
