import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from functools import wraps
from pathlib import Path
from typing import Any, ClassVar, Generic, Self, TypeVar

import pandas as pd
from pydantic import field_serializer
from typing_extensions import override

from lion.core.typing import (
    ID,
    UNDEFINED,
    Field,
    FieldInfo,
    ItemExistsError,
    ItemNotFoundError,
    Observable,
)
from lion.libs.parse import is_same_dtype, to_list
from lion.protocols.adapters.adapter import Adapter, AdapterRegistry
from lion.protocols.registries._pile_registry import PileAdapterRegistry

from .element import Element
from .progression import Progression
from .utils import to_list_type, validate_order

T = TypeVar("T", bound=Element)
D = TypeVar("D")


def synchronized(func: Callable):
    @wraps(func)
    def wrapper(self: "Pile", *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


def async_synchronized(func: Callable):
    @wraps(func)
    async def wrapper(self: "Pile", *args, **kwargs):
        async with self.async_lock:
            return await func(self, *args, **kwargs)

    return wrapper


class Pile(Element, Generic[T]):
    """thread-safe async-compatible, ordered collection of elements."""

    pile_: dict[str, T] = Field(default_factory=dict)
    item_type: set[type[T]] | None = Field(
        default=None,
        description="Set of allowed types for items in the pile.",
        exclude=True,
    )
    progress: Progression = Field(
        default_factory=Progression,
        description="Progression specifying the order of items in the pile.",
        exclude=True,
    )
    strict_type: bool = Field(
        default=False,
        description="Specify if enforce a strict type check",
        frozen=True,
    )

    _adapter_registry: ClassVar[AdapterRegistry] = PileAdapterRegistry

    def __pydantic_extra__(self) -> dict[str, FieldInfo]:
        return {
            "_lock": Field(default_factory=threading.Lock),
            "_async": Field(default_factory=asyncio.Lock),
        }

    def __pydantic_private__(self) -> dict[str, FieldInfo]:
        return self.__pydantic_extra__()

    @override
    def __init__(
        self,
        items: ID.ItemSeq = None,
        item_type: set[type[T]] = None,
        order: ID.RefSeq = None,
        strict_type: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize a Pile instance.

        Args:
            items: Initial items for the pile.
            item_type: Allowed types for items in the pile.
            order: Initial order of items (as Progression).
            strict: If True, enforce strict type checking.
        """
        _config = {}
        if "ln_id" in kwargs:
            _config["ln_id"] = kwargs["ln_id"]
        if "created" in kwargs:
            _config["created"] = kwargs["created"]

        super().__init__(strict_type=strict_type, **_config)
        self.item_type = self._validate_item_type(item_type)
        self.pile_ = self._validate_pile(items or kwargs.get("pile_", {}))
        self.progress = self._validate_order(order)

    # Sync Interface methods
    @override
    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        /,
    ) -> "Pile":
        """Create a Pile instance from a dictionary.

        Args:
            data: A dictionary containing Pile data.

        Returns:
            A new Pile instance created from the provided data.

        Raises:
            ValueError: If the dictionary format is invalid.
        """
        items = data.pop("pile_", [])
        items = [Element.from_dict(i) for i in items]
        return cls(items=items, **data)

    def __setitem__(
        self,
        key: ID.Ref | ID.RefSeq | int | slice,
        item: ID.ItemSeq | ID.Item,
    ) -> None:
        """Set an item or items in the Pile.

        Args:
            key: The key to set. Can be an integer index, a string ID, or a
                slice.
            item: The item or items to set. Must be of type T or a sequence
                of T for slices.

        Raises:
            TypeError: If the item type is not allowed.
            KeyError: If the key is invalid.
            ValueError: If trying to set multiple items with a non-slice key.
        """
        self._setitem(key, item)

    @synchronized
    def pop(
        self,
        key: ID.Ref | ID.RefSeq | int | slice,
        default: D = UNDEFINED,
        /,
    ) -> T | "Pile" | D:
        """Remove and return an item or items from the Pile.

        Args:
            key: The key of the item(s) to remove. Can be an integer index,
                a string ID, or a slice.
            default: The value to return if the key is not found. Defaults to
                UNDEFINED.

        Returns:
            The removed item(s), or the default value if not found.

        Raises:
            KeyError: If the key is not found and no default is provided.
        """
        return self._pop(key, default)

    def remove(
        self,
        item: T,
        /,
    ) -> None:
        """Remove a specific item from the Pile.

        Args:
            item: The item to remove.

        Raises:
            ValueError: If the item is not found in the Pile.
        """
        self._remove(item)

    def include(
        self,
        item: ID.ItemSeq | ID.Item,
        /,
    ) -> None:
        """Include item(s) in the Pile if not already present.

        Args:
            item: Item or iterable of items to include.

        Raises:
            TypeError: If the item(s) are not of allowed types.
        """
        self._include(item)

    def exclude(
        self,
        item: ID.ItemSeq | ID.Item,
        /,
    ) -> None:
        """Exclude item(s) from the Pile if present.

        Args:
            item: Item or iterable of items to exclude.

        Note:
            This method does not raise an error if an item is not found.
        """
        self._exclude(item)

    @synchronized
    def clear(self) -> None:
        """Remove all items from the Pile."""
        self._clear()

    def update(
        self,
        other: ID.Item | ID.ItemSeq,
        /,
    ) -> None:
        """Update Pile with items from another iterable or Pile.

        Args:
            other: An iterable or another Pile to update from.

        Raises:
            TypeError: If the items in 'other' are not of allowed types.
        """
        self._update(other)

    @synchronized
    def insert(self, index: int, item: T, /) -> None:
        """Insert an item at a specific position in the Pile.

        Args:
            index: The index at which to insert the item.
            item: The item to insert.

        Raises:
            IndexError: If the index is out of range.
            TypeError: If the item is not of an allowed type.
        """
        self._insert(index, item)

    @synchronized
    def append(self, item: T, /) -> None:
        """Append an item to the end of the Pile.

        This method is an alias for `include`.

        Args:
            item: The item to append.

        Raises:
            TypeError: If the item is not of an allowed type.
        """
        self.update(item)

    @synchronized
    def get(
        self,
        key: ID.Ref | ID.RefSeq | int | slice,
        default: D = UNDEFINED,
        /,
    ) -> T | "Pile" | D:
        """Retrieve item(s) associated with the given key.

        Args:
            key: The key of item(s) to retrieve. Can be an integer index,
                a string ID, or a slice.
            default: Value to return if the key is not found. Defaults to
                UNDEFINED.

        Returns:
            The item(s) associated with the key, or the default value if not
            found. Returns a new Pile instance for slice keys.

        Note:
            Unlike `__getitem__`, this method does not raise KeyError for
            missing keys when a default is provided.
        """
        return self._get(key, default)

    def keys(self) -> Sequence[str]:
        """Return a sequence of all keys (Lion IDs) in the Pile.

        Returns:
            A sequence of string keys representing the Lion IDs of items
            in their current order.
        """
        return list(self.progress)

    def values(self) -> Sequence[T]:
        """Return a sequence of all values in the Pile.

        Returns:
            A sequence of all items in the Pile in their current order.
        """
        return [self.pile_[key] for key in self.progress]

    def items(self) -> Sequence[tuple[str, T]]:
        """Return a sequence of all (key, value) pairs in the Pile.

        Returns:
            A sequence of tuples, each containing a string key (Lion ID)
            and its corresponding item, in their current order.
        """
        return [(key, self.pile_[key]) for key in self.progress]

    def is_empty(self) -> bool:
        """Check if the Pile is empty.

        Returns:
            True if the Pile contains no items, False otherwise.
        """

        return len(self.progress) == 0

    def size(self) -> int:
        """Get the number of items in the Pile.

        Returns:
            The count of items currently in the Pile.

        Note:
            This method is equivalent to using the `len()` function
            on the Pile.
        """
        return len(self.progress)

    def __iter__(self) -> Iterator[T]:
        """Return an iterator over the items in the Pile.

        This method creates a snapshot of the current order to prevent
        issues with concurrent modifications during iteration.

        Yields:
            Items in the Pile in their current order.
        """
        with self.lock:
            current_order = list(self.progress)

        for key in current_order:
            yield self.pile_[key]

    def __next__(self) -> T:
        """Return the next item in the Pile.

        Returns:
            The next item in the Pile.

        Raises:
            StopIteration: When there are no more items in the Pile.
        """
        try:
            return next(iter(self))
        except StopIteration:
            raise StopIteration("End of pile")

    def __getitem__(self, key: ID.Ref | ID.RefSeq | int | slice) -> Any | list | T:
        """Get item(s) from the Pile by index, ID, or slice.

        Args:
            key: Integer index, string ID, or slice.

        Returns:
            The item or a new Pile containing the sliced items.

        Raises:
            KeyError: If the key is not found.
            TypeError: If the key type is invalid.
        """
        return self._getitem(key)

    def __contains__(self, item: ID.RefSeq | ID.Ref) -> bool:
        """Check if an item is in the Pile.

        Args:
            item: The item to check for.

        Returns:
            True if the item is in the Pile, False otherwise.
        """
        return item in self.progress

    def __len__(self) -> int:
        """Return the number of items in the Pile.

        Returns:
            The number of items in the Pile.
        """
        return len(self.pile_)

    @override
    def __bool__(self) -> bool:
        """Check if the Pile is not empty.

        Returns:
            True if the Pile is not empty, False otherwise.
        """
        return not self.is_empty()

    def __list__(self) -> list[T]:
        """Convert the Pile to a list.

        Returns:
            A list containing all items in the Pile.
        """
        return self.values()

    def __ior__(self, other: "Pile") -> Self:
        if not isinstance(other, Pile):
            raise TypeError(
                "Invalid type for Pile operation.",
                expected_type=Pile,
                actual_type=type(other),
            )
        other = self._validate_pile(list(other))
        self.include(other)
        return self

    def __or__(self, other: "Pile") -> "Pile":
        if not isinstance(other, Pile):
            raise TypeError(
                "Invalid type for Pile operation.",
                expected_type=Pile,
                actual_type=type(other),
            )

        result = self.__class__(
            items=self.values(),
            item_type=self.item_type,
            order=self.progress,
        )
        result.include(list(other))
        return result

    def __ixor__(self, other: "Pile") -> Self:
        if not isinstance(other, Pile):
            raise TypeError(
                "Invalid type for Pile operation.",
                expected_type=Pile,
                actual_type=type(other),
            )

        to_exclude = []
        for i in other:
            if i in self:
                to_exclude.append(i)

        other = [i for i in other if i not in to_exclude]
        self.exclude(to_exclude)
        self.include(other)
        return self

    def __xor__(self, other: "Pile") -> "Pile":
        if not isinstance(other, Pile):
            raise TypeError(
                "Invalid type for Pile operation.",
                expected_type=Pile,
                actual_type=type(other),
            )

        to_exclude = []
        for i in other:
            if i in self:
                to_exclude.append(i)

        values = [i for i in self if i not in to_exclude] + [
            i for i in other if i not in to_exclude
        ]

        result = self.__class__(
            items=values,
            item_type=self.item_type,
        )
        return result

    def __iand__(self, other: "Pile") -> Self:
        if not isinstance(other, Pile):
            raise TypeError(
                "Invalid type for Pile operation.",
                expected_type=Pile,
                actual_type=type(other),
            )

        to_exclude = []
        for i in self.values():
            if i not in other:
                to_exclude.append(i)
        self.exclude(to_exclude)
        return self

    def __and__(self, other: "Pile") -> "Pile":
        if not isinstance(other, Pile):
            raise TypeError(
                "Invalid type for Pile operation.",
                expected_type=Pile,
                actual_type=type(other),
            )

        values = [i for i in self if i in other]
        return self.__class__(
            items=values,
            item_type=self.item_type,
        )

    @override
    def __str__(self) -> str:
        """Return a string representation of the Pile.

        Returns:
            A string in the format "Pile(length)".
        """
        return f"Pile({len(self)})"

    @override
    def __repr__(self) -> str:
        """Return a detailed string representation of the Pile.

        Returns:
            A string representation of the Pile, showing its contents
            for small Piles or just the length for larger ones.
        """
        length = len(self)
        if length == 0:
            return "Pile()"
        elif length == 1:
            return f"Pile({next(iter(self.pile_.values())).__repr__()})"
        else:
            return f"Pile({length})"

    def __getstate__(self):
        """Prepare the Pile instance for pickling."""
        state = self.__dict__.copy()
        state["_lock"] = None
        state["_async_lock"] = None
        return state

    def __setstate__(self, state):
        """Restore the Pile instance after unpickling."""
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    @property
    def lock(self):
        """Ensure the lock is always available, even during unpickling."""
        if not hasattr(self, "_lock") or self._lock is None:
            self._lock = threading.Lock()
        return self._lock

    @property
    def async_lock(self):
        """Ensure the async lock is always available, even during unpickling"""
        if not hasattr(self, "_async_lock") or self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    # Async Interface methods
    @async_synchronized
    async def asetitem(
        self,
        key: ID.Ref | ID.RefSeq | int | slice,
        item: ID.Item | ID.ItemSeq,
        /,
    ) -> None:
        """Asynchronously set an item or items in the Pile.

        Args:
            key: The key to set. Can be an integer index, a string ID, or a
                slice.
            item: The item or items to set. Must be of type T or an iterable
                of T for slices.

        Raises:
            TypeError: If the item type is not allowed.
            KeyError: If the key is invalid.
            ValueError: If trying to set multiple items with a non-slice key.
        """
        self._setitem(key, item)

    @async_synchronized
    async def apop(
        self,
        key: ID.Ref | ID.RefSeq | int | slice,
        default: Any = UNDEFINED,
        /,
    ):
        """Asynchronously remove and return an item or items from the Pile.

        Args:
            key: The key of the item(s) to remove. Can be an integer index,
                a string ID, or a slice.
            default: The value to return if the key is not found. Defaults to
                UNDEFINED.

        Returns:
            The removed item(s), or the default value if not found.

        Raises:
            KeyError: If the key is not found and no default is provided.
        """
        return self._pop(key, default)

    @async_synchronized
    async def aremove(
        self,
        item: ID.Ref | ID.RefSeq,
        /,
    ) -> None:
        """Asynchronously remove a specific item from the Pile.

        Args:
            item: The item to remove.

        Raises:
            ValueError: If the item is not found in the Pile.
        """
        self._remove(item)

    @async_synchronized
    async def ainclude(
        self,
        item: ID.ItemSeq | ID.Item,
        /,
    ) -> None:
        """Asynchronously include item(s) in the Pile if not already present.

        Args:
            item: Item or iterable of items to include.

        Raises:
            TypeError: If the item(s) are not of allowed types.
        """
        self._include(item)
        if item not in self:
            raise TypeError(f"Item {item} is not of allowed types")

    @async_synchronized
    async def aexclude(
        self,
        item: ID.Ref | ID.RefSeq,
        /,
    ) -> None:
        """Asynchronously exclude item(s) from the Pile if present.

        Args:
            item: Item or iterable of items to exclude.

        Note:
            This method does not raise an error if an item is not found.
        """
        self._exclude(item)

    @async_synchronized
    async def aclear(self) -> None:
        self._clear()

    @async_synchronized
    async def aupdate(
        self,
        other: ID.ItemSeq | ID.Item,
        /,
    ) -> None:
        self._update(other)

    @async_synchronized
    async def aget(
        self,
        key: Any,
        default=UNDEFINED,
        /,
    ) -> list | Any | T:
        return self._get(key, default)

    async def __aiter__(self) -> AsyncIterator[T]:
        """Return an asynchronous iterator over the items in the Pile.

        This method creates a snapshot of the current order to prevent
        issues with concurrent modifications during iteration.

        Yields:
            Items in the Pile in their current order.

        Note:
            This method yields control to the event loop after each item,
            allowing other async operations to run between iterations.
        """

        async with self.async_lock:
            current_order = list(self.progress)

        for key in current_order:
            yield self.pile_[key]
            await asyncio.sleep(0)  # Yield control to the event loop

    async def __anext__(self) -> T:
        """Asynchronously return the next item in the Pile."""
        try:
            return await anext(self.AsyncPileIterator(self))
        except StopAsyncIteration:
            raise StopAsyncIteration("End of pile")

    # private methods
    def _getitem(self, key: Any) -> Any | list | T:
        if key is None:
            raise ValueError("getitem key not provided.")

        if isinstance(key, int | slice):
            try:
                result_ids = self.progress[key]
                result_ids = (
                    [result_ids] if not isinstance(result_ids, list) else result_ids
                )
                result = []
                for i in result_ids:
                    result.append(self.pile_[i])
                return result[0] if len(result) == 1 else result
            except Exception as e:
                raise ItemNotFoundError(f"index {key}. Error: {e}")

        elif isinstance(key, str):
            try:
                return self.pile_[key]
            except Exception as e:
                raise ItemNotFoundError(f"key {key}. Error: {e}")

        else:
            key = to_list_type(key)
            result = []
            try:
                for k in key:
                    result_id = ID.get_id(k)
                    result.append(self.pile_[result_id])

                if len(result) == 0:
                    raise ItemNotFoundError(f"key {key} item not found")
                if len(result) == 1:
                    return result[0]
                return result
            except Exception as e:
                raise ItemNotFoundError(f"Key {key}. Error:{e}")

    def _setitem(
        self,
        key: ID.Ref | ID.RefSeq | int | slice,
        item: ID.Item | ID.ItemSeq,
    ) -> None:
        """
        Set new values in the pile using various key types.

        Handles single/multiple assignments, ensures type consistency.
        Supports index/slice, LionID, and LionIDable key access.

        Args:
            key: Key to set items. Can be index, slice, LionID, LionIDable.
            item: Item(s) to set. Can be single item or collection.

        Raises:
            ValueError: Length mismatch or multiple items to single key.
            TypeError: Item type doesn't match allowed types.
        """
        item_dict = self._validate_pile(item)

        item_order = []
        for i in item_dict.keys():
            if i in self.progress:
                raise ItemExistsError(f"item {i} already exists in the pile")
            item_order.append(i)
        if isinstance(key, int | slice):
            try:
                delete_order = (
                    list(self.progress[key])
                    if isinstance(self.progress[key], Progression)
                    else [self.progress[key]]
                )
                self.progress[key] = item_order
                for i in to_list(delete_order, flatten=True):
                    self.pile_.pop(i)
                self.pile_.update(item_dict)
            except Exception as e:
                raise ValueError(f"Failed to set pile. Error: {e}")
        else:
            key = to_list_type(key)
            if isinstance(key[0], list):
                key = to_list(key, flatten=True, dropna=True)
            if len(key) != len(item_order):
                raise KeyError(
                    f"Invalid key {key}. Key and item does not match.",
                )
            for k in key:
                id_ = ID.get_id(k)
                if id_ not in item_order:
                    raise KeyError(
                        f"Invalid key {id_}. Key and item does not match.",
                    )
            self.progress += key
            self.pile_.update(item_dict)

    def _get(self, key: Any, default: D = UNDEFINED) -> T | "Pile" | D:
        """
        Retrieve item(s) associated with given key.

        Args:
            key: Key of item(s) to retrieve. Can be single or collection.
            default: Default value if key not found.

        Returns:
            Retrieved item(s) or default if key not found.

        Raises:
            ItemNotFoundError: If key not found and no default specified.
        """
        if isinstance(key, int | slice):
            try:
                return self[key]
            except Exception as e:
                if default is UNDEFINED:
                    raise ItemNotFoundError(f"Item not found. Error: {e}")
                return default
        else:
            check = None
            if isinstance(key, list):
                check = True
                for i in key:
                    if type(i) is not int:
                        check = False
                        break
            try:
                if not check:
                    key = validate_order(key)
                result = []
                for k in key:
                    result.append(self[k])
                if len(result) == 0:
                    raise ItemNotFoundError(f"key {key} item not found")
                if len(result) == 1:
                    return result[0]
                return result

            except Exception as e:
                if default is UNDEFINED:
                    raise ItemNotFoundError(f"Item not found. Error: {e}")
                return default

    def _pop(
        self,
        key: ID.Ref | ID.RefSeq | int | slice,
        default: D = UNDEFINED,
    ) -> T | "Pile" | D:
        """
        Remove and return item(s) associated with given key.

        Args:
            key: Key of item(s) to remove. Can be single or collection.
            default: Default value if key not found.

        Returns:
            Removed item(s) or default if key not found.

        Raises:
            ItemNotFoundError: If key not found and no default specified.
        """
        if isinstance(key, int | slice):
            try:
                pops = self.progress[key]
                pops = [pops] if isinstance(pops, str) else pops
                result = []
                for i in pops:
                    self.progress.remove(i)
                    result.append(self.pile_.pop(i))
                result = (
                    self.__class__(items=result, item_type=self.item_type)
                    if len(result) > 1
                    else result[0]
                )
                return result
            except Exception as e:
                if default is UNDEFINED:
                    raise ItemNotFoundError(f"Item not found. Error: {e}")
                return default
        else:
            try:
                key = validate_order(key)
                result = []
                for k in key:
                    self.progress.remove(k)
                    result.append(self.pile_.pop(k))
                if len(result) == 0:
                    raise ItemNotFoundError(f"key {key} item not found")
                elif len(result) == 1:
                    return result[0]
                return result
            except Exception as e:
                if default is UNDEFINED:
                    raise ItemNotFoundError(f"Item not found. Error: {e}")
                return default

    def _remove(self, item: ID.Ref | ID.RefSeq):
        """
        Remove an item from the pile.

        Args:
            item: The item to remove.

        Raises:
            ItemNotFoundError: If the item is not found in the pile.
        """
        if isinstance(item, int | slice):
            raise TypeError("Invalid item type for remove, should be ID or Item(s)")
        if item in self:
            self.pop(item)
            return
        raise ItemNotFoundError(f"{item}")

    def _include(self, item: ID.ItemSeq | ID.Item):
        """
        Include item(s) in pile if not already present.

        Args:
            item: Item(s) to include. Can be single item or collection.
        """
        item_dict = self._validate_pile(item)

        item_order = []
        for i in item_dict.keys():
            if i not in self.progress:
                item_order.append(i)

        self.progress.append(item_order)
        self.pile_.update(item_dict)

    def _exclude(self, item: ID.Ref | ID.RefSeq):
        """
        Exclude item(s) from pile if present.

        Args:
            item: Item(s) to exclude. Can be single item or collection.
        """
        item = to_list_type(item)
        exclude_list = []
        for i in item:
            if i in self:
                exclude_list.append(i)
        if exclude_list:
            self.pop(exclude_list)

    def _clear(self) -> None:
        """Remove all items from the pile."""
        self.pile_.clear()
        self.progress.clear()

    def _update(self, other: ID.ItemSeq | ID.Item):
        """Update pile with another collection of items."""
        others = self._validate_pile(other)
        for i in others.keys():
            if i in self.pile_:
                self.pile_[i] = others[i]
            else:
                self.include(others[i])

    def _validate_item_type(self, value) -> set[type[T]] | None:
        """
        Validate the item type for the pile.

        Ensures that the provided item type is a subclass of Element or iModel.
        Raises an error if the validation fails.

        Args:
            value: The item type to validate. Can be a single type or a list of
                    types.

        Returns:
            set: A set of validated item types.

        Raises:
            TypeError: If an invalid item type is provided.
            ValueError: If duplicate item types are detected.
        """
        if value is None:
            return None

        value = to_list_type(value)

        for i in value:
            if not issubclass(i, Observable):
                raise TypeError(
                    message="Item type must be a subclass of T.",
                    expected_type=T,
                    actual_type=type(i),
                )

        if len(value) != len(set(value)):
            raise ValueError(
                "Detected duplicated item types in item_type.",
            )

        if len(value) > 0:
            return set(value)

    def _validate_pile(self, value: Any) -> dict[str, T]:
        """Validate and convert the items to be added to the pile."""
        if not value:
            return {}

        value = to_list_type(value)

        result = {}
        for i in value:
            if self.item_type:
                if self.strict_type:
                    if type(i) not in self.item_type:
                        raise TypeError(
                            message="Invalid item type in pile."
                            f" Expected {self.item_type}",
                        )
                else:
                    if not any(issubclass(type(i), t) for t in self.item_type):
                        raise TypeError(
                            "Invalid item type in pile. Expected "
                            f"{self.item_type} or the subclasses",
                        )
            else:
                if not isinstance(i, Observable):
                    raise ValueError(f"Invalid pile item {i}")

            result[i.ln_id] = i

        return result

    def _validate_order(self, value: Any) -> Progression:
        if not value:
            return self.progress.__class__(order=list(self.pile_.keys()))

        if isinstance(value, Progression):
            value = list(value)
        else:
            value = to_list_type(value)

        value_set = set(value)
        if len(value_set) != len(value):
            raise ValueError("There are duplicate elements in the order")
        if len(value_set) != len(self.pile_.keys()):
            raise ValueError(
                "The length of the order does not match the length of the pile"
            )

        for i in value_set:
            if ID.get_id(i) not in self.pile_.keys():
                raise ValueError(f"The order does not match the pile. {i} not found")

        return self.progress.__class__(order=value)

    def _insert(self, index: int, item: ID.Item):
        item_dict = self._validate_pile(item)

        item_order = []
        for i in item_dict.keys():
            if i in self.progress:
                raise ItemExistsError(f"item {i} already exists in the pile")
            item_order.append(i)
        self.progress.insert(index, item_order)
        self.pile_.update(item_dict)

    @field_serializer("pile_")
    def _(self, value: dict[str, T]):
        return [i.to_dict() for i in value.values()]

    class AsyncPileIterator:
        def __init__(self, pile: "Pile"):
            self.pile = pile
            self.index = 0

        def __aiter__(self) -> AsyncIterator[T]:
            return self

        async def __anext__(self) -> T:
            if self.index >= len(self.pile):
                raise StopAsyncIteration
            item = self.pile[self.pile.progress[self.index]]
            self.index += 1
            await asyncio.sleep(0)  # Yield control to the event loop
            return item

    async def __aenter__(self) -> Self:
        """
        Enter async context - useful for bulk operations that need
        guaranteed cleanup.
        """
        await self.async_lock.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """
        Exit async context - ensures lock release.
        """
        self.async_lock.release()

    def is_homogenous(self) -> bool:
        return len(self.pile_) < 2 or all(is_same_dtype(self.pile_.values()))

    def adapt_to(self, obj_key: str, /, **kwargs: Any) -> Any:
        return self._get_adapter_registry().adapt_to(self, obj_key, **kwargs)

    @classmethod
    def list_adapters(cls):
        return cls._get_adapter_registry().list_adapters()

    @classmethod
    def register_adapter(cls, adapter: type[Adapter]):
        cls._get_adapter_registry().register(adapter)

    @classmethod
    def _get_adapter_registry(cls) -> AdapterRegistry:
        """Get the converter registry for the class."""
        if isinstance(cls._adapter_registry, type):
            cls._adapter_registry = cls._adapter_registry()
        return cls._adapter_registry

    @classmethod
    def adapt_from(cls, obj: Any, obj_key: str, /, **kwargs: Any):
        dict_ = cls._get_adapter_registry().adapt_from(cls, obj, obj_key, **kwargs)
        if isinstance(dict_, list):
            dict_ = {"pile_": dict_}
        return cls.from_dict(dict_)

    def to_df(
        self,
        columns: list[str] | None = None,
        **kwargs: Any,
    ):
        return self.adapt_to("pd_dataframe", columns=columns, **kwargs)

    def to_csv(self, fp: str | Path, **kwargs: Any) -> None:
        self.adapt_to(".csv", fp=fp, **kwargs)

    def to_excel(self, fp: str | Path, **kwargs: Any) -> None:
        self.adapt_to(".xlsx", fp=fp, **kwargs)


def pile(
    items: Any = None,
    /,
    item_type: type[T] | set[type[T]] | None = None,
    order: list[str] | None = None,
    strict_type: bool = False,
    df: pd.DataFrame | None = None,  # priority 1
    fp: str | Path | None = None,  # priority 2
    **kwargs,
) -> Pile:
    """
    Create a new Pile instance.

    Args:
        items: Initial items for the pile.
        item_type: Allowed types for items in the pile.
        order: Initial order of items.
        strict: If True, enforce strict type checking.

    Returns:
        Pile: A new Pile instance.
    """

    if df:
        return Pile.adapt_from(df, "pd_dataframe", **kwargs)

    if fp:
        fp = Path(fp)
        if fp.suffix == ".csv":
            return Pile.adapt_from(fp, ".csv", **kwargs)
        if fp.suffix == ".xlsx":
            return Pile.adapt_from(fp, ".xlsx", **kwargs)
        if fp.suffix == ".json":
            return Pile.adapt_from(fp, ".json", **kwargs)

    return Pile(
        items,
        item_type=item_type,
        order=order,
        strict=strict_type,
        **kwargs,
    )


__all__ = [Pile, pile]
# File: autoos/generic/pile.py
