"""Tests for the Progression class."""

import pytest
from pydantic import ValidationError

from lion.core.generic import Element, Progression
from lion.core.typing import IDError, ItemNotFoundError


class TestElement(Element):
    """Test element class for progression testing."""

    value: int = None

    def __init__(self, value=None, **kwargs):
        super().__init__(**kwargs)
        self.value = value


@pytest.fixture
def empty_progression():
    """Create an empty progression."""
    return Progression()


@pytest.fixture
def sample_progression():
    """Create a progression with some test elements."""
    elements = [TestElement(i) for i in range(3)]
    return Progression(order=[e.ln_id for e in elements])


def test_progression_creation():
    """Test basic progression creation."""
    prog = Progression()
    assert len(prog) == 0
    assert prog.is_empty()

    # Test with name
    prog = Progression(name="test_progression")
    assert prog.name == "test_progression"


def test_progression_with_order():
    """Test progression creation with initial order."""
    elements = [TestElement() for _ in range(3)]
    ids = [e.ln_id for e in elements]

    prog = Progression(order=ids)
    assert len(prog) == 3
    assert list(prog) == ids


def test_progression_validation():
    """Test progression validation."""
    # Test with invalid ID
    with pytest.raises((ValidationError, IDError)):
        Progression(order=["invalid id with spaces"])

    # Test with duplicate IDs
    element = TestElement()
    prog = Progression(order=[element.ln_id, element.ln_id])
    assert len(prog) == 2  # Duplicates are allowed


def test_progression_contains(sample_progression):
    """Test contains functionality."""
    element = TestElement()

    # Test with ID string
    assert sample_progression.order[0] in sample_progression

    # Test with Element object
    assert element not in sample_progression


def test_progression_indexing(sample_progression):
    """Test indexing operations."""
    # Test getting by index
    first_id = sample_progression[0]
    assert isinstance(first_id, str)

    # Test slicing
    slice_result = sample_progression[0:2]
    assert isinstance(slice_result, Progression)
    assert len(slice_result) == 2

    # Test invalid index
    with pytest.raises(ItemNotFoundError):
        sample_progression[100]


def test_progression_modification(sample_progression):
    """Test modification operations."""
    new_element = TestElement()

    # Test append
    original_len = len(sample_progression)
    sample_progression.append(new_element.ln_id)
    assert len(sample_progression) == original_len + 1
    assert new_element.ln_id in sample_progression

    # Test remove
    sample_progression.remove(new_element.ln_id)
    assert new_element.ln_id not in sample_progression

    # Test clear
    sample_progression.clear()
    assert len(sample_progression) == 0


def test_progression_iteration(sample_progression):
    """Test iteration functionality."""
    ids = list(sample_progression)
    assert len(ids) == 3
    assert all(isinstance(id_, str) for id_ in ids)


def test_progression_operations(sample_progression):
    """Test progression operations."""
    new_element = TestElement()

    # Test include
    sample_progression.include(new_element.ln_id)
    assert new_element.ln_id in sample_progression

    # Test exclude
    sample_progression.exclude(new_element.ln_id)
    assert new_element.ln_id not in sample_progression

    # Test pop
    popped = sample_progression.pop()
    assert isinstance(popped, str)
    assert popped not in sample_progression

    # Test popleft
    leftmost = sample_progression.popleft()
    assert isinstance(leftmost, str)
    assert leftmost not in sample_progression


def test_progression_comparison():
    """Test comparison operations."""
    prog1 = Progression(order=[TestElement().ln_id for _ in range(3)])
    prog2 = Progression(order=prog1.order[:])
    prog3 = Progression(order=[TestElement().ln_id for _ in range(3)])

    assert prog1 == prog2  # Same order
    assert prog1 != prog3  # Different IDs


def test_progression_insert():
    """Test insertion operations."""
    prog = Progression()
    elements = [TestElement() for _ in range(3)]

    # Insert at beginning
    prog.insert(0, elements[0].ln_id)
    assert prog[0] == elements[0].ln_id

    # Insert in middle
    prog.insert(1, elements[1].ln_id)
    assert prog[1] == elements[1].ln_id

    # Insert at end
    prog.insert(2, elements[2].ln_id)
    assert prog[2] == elements[2].ln_id


def test_progression_extend():
    """Test extend functionality."""
    prog1 = Progression(order=[TestElement().ln_id for _ in range(2)])
    prog2 = Progression(order=[TestElement().ln_id for _ in range(2)])

    prog1.extend(prog2)
    assert len(prog1) == 4

    # Test with invalid type
    with pytest.raises(TypeError):
        prog1.extend([])  # Should only accept Progression


def test_progression_count():
    """Test count functionality."""
    element = TestElement()
    prog = Progression(order=[element.ln_id])

    assert prog.count(element.ln_id) == 1
    assert prog.count(TestElement().ln_id) == 0


def test_progression_index():
    """Test index functionality."""
    elements = [TestElement() for _ in range(3)]
    prog = Progression(order=[e.ln_id for e in elements])

    # Test basic index
    assert prog.index(elements[1].ln_id) == 1

    # Test with start parameter
    assert prog.index(elements[2].ln_id, start=1) == 2

    # Test with start and end parameters
    with pytest.raises(ValueError):
        prog.index(elements[2].ln_id, start=0, end=1)


def test_progression_serialization():
    """Test serialization functionality."""
    original = Progression(name="test", order=[TestElement().ln_id for _ in range(3)])

    # Test to_dict
    dict_repr = original.to_dict()
    assert isinstance(dict_repr, dict)
    assert dict_repr["name"] == "test"
    assert len(dict_repr["order"]) == 3

    # Test from_dict
    recreated = Progression.from_dict(dict_repr)
    assert recreated == original


def test_progression_string_representation():
    """Test string representations."""
    prog = Progression(name="test", order=[TestElement().ln_id for _ in range(3)])

    # Test str
    str_repr = str(prog)
    assert "test" in str_repr
    assert "size=3" in str_repr

    # Test repr
    repr_str = repr(prog)
    assert "Progression" in repr_str
    assert all(id_ in repr_str for id_ in prog.order)


def test_progression_bool():
    """Test boolean evaluation."""
    prog = Progression()
    assert not bool(prog)  # Empty progression is False

    prog.append(TestElement().ln_id)
    assert bool(prog)  # Non-empty progression is True


def test_progression_reverse():
    """Test reverse functionality."""
    elements = [TestElement() for _ in range(3)]
    prog = Progression(order=[e.ln_id for e in elements])

    reversed_prog = prog.reverse()  # Use reverse() method instead of __reverse__
    assert list(reversed_prog) == list(reversed(prog.order))
