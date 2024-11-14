"""Tests for the Log class."""

import pytest

from lion.core.generic import Log
from lion.core.typing import Note


def test_log_creation():
    """Test basic Log creation."""
    content = Note(**{"message": "test message"})
    loginfo = Note(**{"level": "INFO"})
    log = Log(content=content, loginfo=loginfo)

    assert isinstance(log.content, Note)
    assert isinstance(log.loginfo, Note)
    assert log.content["message"] == "test message"
    assert log.loginfo["level"] == "INFO"


def test_log_empty_notes():
    """Test Log creation with empty Notes."""
    log = Log(content=Note(), loginfo=Note())
    assert isinstance(log.content, Note)
    assert isinstance(log.loginfo, Note)
    assert len(log.content.content) == 0
    assert len(log.loginfo.content) == 0


def test_log_with_dict_content():
    """Test Log creation with dictionary content."""
    content = {"content": {"message": "test message"}}
    loginfo = {"content": {"level": "INFO"}}
    log = Log(content=Note(**content), loginfo=Note(**loginfo))

    assert isinstance(log.content, Note)
    assert isinstance(log.loginfo, Note)
    assert log.content["content"]["message"] == "test message"
    assert log.loginfo["content"]["level"] == "INFO"


def test_log_immutability():
    """Test Log immutability after creation."""
    log = Log(content=Note(content={"message": "test"}), loginfo=Note())

    # Try to modify after setting immutable
    log._immutable = True
    with pytest.raises(AttributeError):
        log.content = Note(content={"new": "content"})


def test_log_to_dict():
    """Test Log conversion to dictionary."""
    content = Note(content={"message": "test message"})
    loginfo = Note(content={"level": "INFO"})
    log = Log(content=content, loginfo=loginfo)

    dict_repr = log.to_dict()
    assert isinstance(dict_repr, dict)
    assert dict_repr["content"]["content"]["content"]["message"] == "test message"
    assert dict_repr["loginfo"]["content"]["content"]["level"] == "INFO"
    assert "log_id" in dict_repr
    assert "log_timestamp" in dict_repr
    assert "log_class" in dict_repr


def test_log_from_dict():
    """Test Log creation from dictionary."""
    original = Log(
        content=Note(content={"message": "test"}),
        loginfo=Note(content={"level": "INFO"}),
    )
    dict_repr = original.to_dict()
    recreated = Log.from_dict(dict_repr)

    assert recreated.content["content"]["content"]["message"] == "test"
    assert recreated.loginfo["content"]["content"]["level"] == "INFO"
    assert recreated.ln_id == original.ln_id
    assert recreated.timestamp == original.timestamp


def test_log_invalid_load():
    """Test Log creation from invalid dictionary."""
    invalid_dict = {
        # Missing required fields
        "invalid": "data"
    }
    with pytest.raises(ValueError):
        Log.from_dict(invalid_dict)


def test_log_to_note():
    """Test Log conversion to Note."""
    content = Note(content={"message": "test message"})
    loginfo = Note(content={"level": "INFO"})
    log = Log(content=content, loginfo=loginfo)

    note = log.to_note()
    assert isinstance(note, Note)
    assert note.content["content"]["content"]["content"]["message"] == "test message"
    assert note.content["loginfo"]["content"]["content"]["level"] == "INFO"
    assert "log_id" in note.content
    assert "log_timestamp" in note.content
    assert "log_class" in note.content


def test_log_validation():
    """Test Log validation of content and loginfo."""
    # Test with invalid content type
    with pytest.raises(Exception):
        Log(content=123, loginfo=Note())  # Invalid content type

    # Test with invalid loginfo type
    with pytest.raises(Exception):
        Log(content=Note(), loginfo=123)  # Invalid loginfo type


def test_log_empty_creation():
    """Test Log creation with no parameters."""
    with pytest.raises(TypeError):
        Log()  # Should raise error as content and loginfo are required


def test_log_nested_notes():
    """Test Log with nested Note structures."""
    metadata = Note(content={"version": "1.0"})
    details = Note(content={"source": "unit test", "metadata": metadata})
    content = Note(content={"message": "test", "details": details})
    context = Note(content={"module": "test_module"})
    loginfo = Note(content={"level": "INFO", "context": context})

    log = Log(content=content, loginfo=loginfo)
    dict_repr = log.to_dict()

    # Verify content structure
    content_dict = dict_repr["content"]["content"]
    assert content_dict["content"]["message"] == "test"
    assert (
        content_dict["content"]["details"]["content"]["content"]["source"]
        == "unit test"
    )
    assert (
        content_dict["content"]["details"]["content"]["content"]["metadata"]["content"][
            "content"
        ]["version"]
        == "1.0"
    )

    # Verify loginfo structure
    loginfo_dict = dict_repr["loginfo"]["content"]
    assert loginfo_dict["content"]["level"] == "INFO"
    assert (
        loginfo_dict["content"]["context"]["content"]["content"]["module"]
        == "test_module"
    )
