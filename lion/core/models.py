import inspect
from collections.abc import Callable, ItemsView, Iterator, ValuesView
from typing import Any, Self, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    create_model,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from typing_extensions import override

from lion.libs.constants import UNDEFINED
from lion.libs.parse import (
    flatten,
    is_same_dtype,
    nget,
    ninsert,
    npop,
    nset,
    to_list,
    validate_boolean,
)
from lion.libs.utils import copy, unique_hash

INDICE_TYPE = str | list[str | int]
FIELD_NAME = TypeVar("FIELD_NAME", bound=str)


common_config = {
    "populate_by_name": True,
    "arbitrary_types_allowed": True,
    "use_enum_values": True,
}


class BaseAutoModel(BaseModel):

    def clean_dump(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self.model_dump(exclude_unset=True).items()
            if v is not UNDEFINED
        }

    def to_dict(self):
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls.model_validate(data)


class SchemaModel(BaseAutoModel):

    model_config = ConfigDict(extra="forbid", validate_default=False, **common_config)
    _unique_hash: str = PrivateAttr(lambda: unique_hash(32))

    @classmethod
    def keys(cls) -> list[str]:
        return list(cls.model_fields.keys())

    def __hash__(self) -> int:
        return hash(self._unique_hash)


class FieldModel(SchemaModel):

    model_config = ConfigDict(extra="allow", validate_default=False, **common_config)

    default: Any = UNDEFINED
    default_factory: Callable = UNDEFINED
    title: str = UNDEFINED
    description: str = UNDEFINED
    examples: list = UNDEFINED
    validators: list = UNDEFINED
    exclude: bool = UNDEFINED
    deprecated: bool = UNDEFINED
    frozen: bool = UNDEFINED
    alias: str = (UNDEFINED,)
    alias_priority: int = (UNDEFINED,)

    name: str = Field(..., exclude=True)
    annotation: type | Any = Field(UNDEFINED, exclude=True)
    validator: Callable | Any = Field(UNDEFINED, exclude=True)
    validator_kwargs: dict | Any = Field(default_factory=dict, exclude=True)

    @property
    def field_info(self) -> FieldInfo:

        annotation = self.annotation if self.annotation is not UNDEFINED else Any
        field_obj: FieldInfo = Field(**self.clean_dump())  # type: ignore
        field_obj.annotation = annotation
        return field_obj

    @property
    def field_validator(self) -> dict[str, Callable]:
        if self.validator is UNDEFINED:
            return None
        kwargs = self.validator_kwargs or {}
        return {
            f"{self.name}_validator": field_validator(self.name, **kwargs)(
                self.validator
            )
        }


class OperableModel(BaseAutoModel):

    model_config = ConfigDict(extra="forbid", validate_default=False, **common_config)

    extra_fields: dict[str, Any] = Field(default_factory=dict)

    @field_serializer("extra_fields")
    def _serialize_extra_fields(
        self,
        value: dict[str, FieldInfo],
    ) -> dict[str, Any]:
        """Custom serializer for extra fields."""
        output_dict = {}
        for k in value.keys():
            k_value = self.__dict__.get(k)
            if isinstance(k_value, SchemaModel):
                k_value = k_value.clean_dump()
            output_dict[k] = k_value
        return output_dict

    @field_validator("extra_fields")
    def _validate_extra_fields(
        cls,
        value: list[FieldModel] | dict[str, FieldModel | FieldInfo],
    ) -> dict[str, FieldInfo]:
        out = {}
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, FieldModel):
                    out[k] = v.field_info
                elif isinstance(v, FieldInfo):
                    out[k] = v
            return out

        elif isinstance(value, list) and is_same_dtype(value, FieldModel):
            return {v.name: v.field_info for v in value}

        raise ValueError("Invalid extra_fields value")

    def clean_dump(self) -> dict:
        dict_ = self.model_dump()
        dict_.pop("extra_fields")
        dict_.update(self._serialize_extra_fields(self.extra_fields))
        for k, v in dict_.items():
            if v is UNDEFINED:
                dict_.pop(k)
        return dict_

    def to_dict(self) -> dict:
        dict_ = self.model_dump()
        dict_.pop("extra_fields")
        dict_.update(self._serialize_extra_fields(self.extra_fields))
        return dict_

    @property
    def all_fields(self) -> dict[str, FieldInfo]:
        """
        Get all fields including model fields and extra fields.

        Returns:
            dict[str, FieldInfo]: A dictionary containing all fields.
        """
        a = {**self.model_fields, **self.extra_fields}
        a.pop("extra_fields", None)
        return a

    def add_field(
        self,
        field_name: FIELD_NAME,
        /,
        value: Any = UNDEFINED,
        annotation: type = UNDEFINED,
        field_obj: FieldInfo = UNDEFINED,
        field_model: FieldModel = UNDEFINED,
        **kwargs,
    ) -> None:
        """
        Add a new field to the component's extra fields.

        Args:
            field_name: The name of the field to add.
            value: The value of the field.
            annotation: Type annotation for the field.
            field_obj: A pre-configured FieldInfo object.
            **kwargs: Additional keyword arguments for Field configuration.

        Raises:
            ValueError: If the field already exists.
        """
        if field_name in self.all_fields:
            raise ValueError(f"Field '{field_name}' already exists")

        self.update_field(
            field_name,
            value=value,
            annotation=annotation,
            field_obj=field_obj,
            field_model=field_model,
            **kwargs,
        )

    def update_field(
        self,
        field_name: FIELD_NAME,
        /,
        value: Any = UNDEFINED,
        annotation: type = None,
        field_obj: FieldInfo = None,
        field_model: FieldModel = None,
        **kwargs,
    ) -> None:
        """
        Update an existing field or create a new one if it doesn't exist.

        Args:
            field_name: The name of the field to update or create.
            value: The new value for the field.
            annotation: Type annotation for the field.
            field_obj: A pre-configured FieldInfo object.
            **kwargs: Additional keyword arguments for Field configuration.

        Raises:
            ValueError: If both 'default' and 'default_factory' are
                        provided in kwargs.
        """

        # pydanitc Field object cannot have both default and default_factory
        if "default" in kwargs and "default_factory" in kwargs:
            raise ValueError(
                "Cannot provide both 'default' and 'default_factory'",
            )

        if field_obj and field_model:
            raise ValueError(
                "Cannot provide both 'field_obj' and 'field_model'",
            )

        # handle field_obj
        if field_obj:
            if not isinstance(field_obj, FieldInfo):
                raise ValueError(
                    "Invalid field_obj, should be a pydantic FieldInfo object"
                )
            self.extra_fields[field_name] = field_obj

        if field_model:
            if not isinstance(field_model, FieldModel):
                raise ValueError("Invalid field_model, should be a FieldModel object")
            self.extra_fields[field_name] = field_model.field_info

        # handle kwargs
        if kwargs:
            if field_name in self.all_fields:  # existing field
                for k, v in kwargs.items():
                    self.field_setattr(field_name, k, v)
            else:
                self.extra_fields[field_name] = Field(**kwargs)

        # handle no explicit defined field
        if not field_obj and not kwargs:
            if field_name not in self.all_fields:
                self.extra_fields[field_name] = Field()

        field_obj = self.all_fields[field_name]

        # handle annotation
        if annotation is not None:
            field_obj.annotation = annotation
        if not field_obj.annotation:
            field_obj.annotation = Any

        # handle value
        if value is UNDEFINED:
            if getattr(self, field_name, UNDEFINED) is not UNDEFINED:
                value = getattr(self, field_name)

            elif getattr(field_obj, "default") is not PydanticUndefined:
                value = field_obj.default

            elif getattr(field_obj, "default_factory"):
                value = field_obj.default_factory()

        setattr(self, field_name, value)

    # field management methods
    def field_setattr(
        self,
        field_name: FIELD_NAME,
        attr: str,
        value: Any,
        /,
    ) -> None:
        """Set the value of a field attribute."""
        all_fields = self.all_fields
        if field_name not in all_fields:
            raise KeyError(f"Field {field_name} not found in object fields.")
        field_obj = all_fields[field_name]
        if hasattr(field_obj, attr):
            setattr(field_obj, attr, value)
        else:
            if not isinstance(field_obj.json_schema_extra, dict):
                field_obj.json_schema_extra = {}
            field_obj.json_schema_extra[attr] = value

    def field_hasattr(
        self,
        field_name: FIELD_NAME,
        attr: str,
        /,
    ) -> bool:
        """Check if a field has a specific attribute."""
        all_fields = self.all_fields
        if field_name not in all_fields:
            raise KeyError(f"Field {field_name} not found in object fields.")
        field_obj = all_fields[field_name]
        if hasattr(field_obj, attr):
            return True
        elif isinstance(field_obj.json_schema_extra, dict):
            if field_name in field_obj.json_schema_extra:
                return True
        else:
            return False

    def field_getattr(
        self,
        field_name: FIELD_NAME,
        attr: str,
        default: Any = UNDEFINED,
        /,
    ) -> Any:
        """Get the value of a field attribute."""
        all_fields = self.all_fields

        if field_name not in all_fields:
            raise KeyError(f"Field {field_name} not found in object fields.")

        if str(attr).strip("s").lower() == "annotation":
            return self.model_fields[field_name].annotation

        field_obj = all_fields[field_name]

        # check fieldinfo attr
        value = getattr(field_obj, attr, UNDEFINED)
        if value is not UNDEFINED:
            return value
        else:
            if isinstance(field_obj.json_schema_extra, dict):
                value = field_obj.json_schema_extra.get(attr, UNDEFINED)
                if value is not UNDEFINED:
                    return value

        # undefined attr
        if default is not UNDEFINED:
            return default
        else:
            raise AttributeError(
                f"field {field_name} has no attribute {attr}",
            )


class Note(BaseAutoModel):
    """A container for managing nested dictionary data structures."""

    content: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        populate_by_name=True,
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a Note instance with the given keyword arguments."""
        super().__init__()
        self.content = kwargs

    @field_serializer("content")
    def _serialize_content(self, value: Any) -> dict[str, Any]:
        """Serialize the content"""
        output_dict = copy(value, deep=True)
        return output_dict

    def clean_dump(self) -> dict[str, Any]:
        out = copy(self.content)

        for k, v in self.content.items():
            if v is UNDEFINED:
                out.pop(k)
        return out

    def pop(
        self,
        indices: INDICE_TYPE,
        /,
        default: Any = UNDEFINED,
    ) -> Any:
        """Remove and return an item from the nested structure."""
        indices = to_list(indices, flatten=True, dropna=True)
        return npop(self.content, indices, default)

    def insert(self, indices: INDICE_TYPE, value: Any, /) -> None:
        """Insert a value into the nested structure at the specified indice"""
        indices = to_list(indices, flatten=True, dropna=True)
        ninsert(self.content, indices, value)

    def set(self, indices: INDICE_TYPE, value: Any, /) -> None:
        """Set a value in the nested structure at the specified indice"""
        indices = to_list(indices, flatten=True, dropna=True)

        if self.get(indices, None) is None:
            self.insert(indices, value)
        else:
            nset(self.content, indices, value)

    def get(
        self,
        indices: INDICE_TYPE,
        /,
        default: Any = UNDEFINED,
    ) -> Any:
        """Get a value from the nested structure at the specified indice"""
        indices = to_list(indices, flatten=True, dropna=True)
        return nget(self.content, indices, default)

    def keys(self, /, flat: bool = False, **kwargs: Any) -> list:
        """
        Get the keys of the Note.

        Args:
            flat: If True, return flattened keys.
            kwargs: Additional keyword arguments for flattening
        """
        if flat:
            kwargs["coerce_keys"] = kwargs.get("coerce_keys", False)
            kwargs["coerce_sequence"] = kwargs.get("coerce_sequence", "list")
            return flatten(self.content, **kwargs).keys()
        return list(self.content.keys())

    def values(self, /, flat: bool = False, **kwargs: Any) -> ValuesView:
        """
        Get the values of the Note.

        Args:
            flat: If True, return flattened values.
            kwargs: Additional keyword arguments for flattening
        """
        if flat:
            kwargs["coerce_keys"] = kwargs.get("coerce_keys", False)
            kwargs["coerce_sequence"] = kwargs.get("coerce_sequence", "list")
            return flatten(self.content, **kwargs).values()
        return self.content.values()

    def items(self, /, flat: bool = False, **kwargs: Any) -> ItemsView:
        """
        Get the items of the Note.

        Args:
            flat: If True, return flattened items.
            kwargs: Additional keyword arguments for flattening
        """
        if flat:
            kwargs["coerce_keys"] = kwargs.get("coerce_keys", False)
            kwargs["coerce_sequence"] = kwargs.get("coerce_sequence", "list")
            return flatten(self.content, **kwargs).items()
        return self.content.items()

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        """
        Convert the Note to a dictionary.

        kwargs: Additional keyword arguments for BaseModel.model_dump

        Returns:
            A dictionary representation of the Note.
        """
        output_dict = self.model_dump(**kwargs)
        return output_dict["content"]

    def clear(self) -> None:
        """Clear the content of the Note."""
        self.content.clear()

    def update(
        self,
        indices: INDICE_TYPE,
        value: Any,
    ) -> None:
        existing = None
        if not indices:
            existing = self.content
        else:
            existing = self.get(indices, None)

        if existing is None:
            if not isinstance(value, (list, dict)):
                value = [value]
            self.set(indices, value)

        if isinstance(existing, list):
            if isinstance(value, list):
                existing.extend(value)
            else:
                existing.append(value)

        elif isinstance(existing, dict):
            if isinstance(value, self.__class__):
                value = value.content

            if isinstance(value, dict):
                existing.update(value)
            else:
                raise ValueError(
                    "Cannot update a dictionary with a non-dictionary value."
                )

    @classmethod
    def from_dict(cls, kwargs: Any) -> "Note":
        """Create a Note from a dictionary."""
        return cls(**kwargs)

    def __contains__(self, indices: INDICE_TYPE) -> bool:
        """Check if the Note contains the specified indices."""
        return self.content.get(indices, UNDEFINED) is not UNDEFINED

    def __len__(self) -> int:
        """Return the length of the Note's content."""
        return len(self.content)

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the Note's content."""
        return iter(self.content)

    def __next__(self) -> str:
        """Return the next item in the Note's content."""
        return next(iter(self.content))

    @override
    def __str__(self) -> str:
        """Return a string representation of the Note's content."""
        return str(self.content)

    @override
    def __repr__(self) -> str:
        """Return a detailed string representation of the Note's content."""
        return repr(self.content)

    def __getitem__(self, indices: INDICE_TYPE) -> Any:
        """Get an item from the Note using index notation."""
        indices = to_list(indices, flatten=True, dropna=True)
        return self.get(indices)

    def __setitem__(self, indices: INDICE_TYPE, value: Any) -> None:
        """Set an item in the Note using index notation."""
        self.set(indices, value)


class NewModelParams(SchemaModel):

    name: str | None = None
    parameter_fields: dict[str, FieldInfo] = Field(default_factory=dict)
    base_type: type[BaseModel] = Field(default=BaseModel)
    field_models: list[FieldModel] = Field(default_factory=list)
    exclude_fields: list = Field(default_factory=list)
    field_descriptions: dict = Field(default_factory=dict)
    inherit_base: bool = Field(default=True)
    use_base_kwargs: bool = False
    config_dict: dict | None = Field(default=None)
    doc: str | None = Field(default=None)
    _class_kwargs: dict = PrivateAttr(default_factory=dict)
    frozen: bool = False
    _validators: dict[str, Callable] | None = PrivateAttr(default=None)
    _use_keys: set[str] = PrivateAttr(default_factory=set)

    @property
    def use_fields(self):
        params = {k: v for k, v in self.parameter_fields.items() if k in self._use_keys}
        params.update(
            {
                f.name: f.field_info
                for f in self.field_models
                if f.name in self._use_keys
            }
        )
        return {k: (v.annotation, v) for k, v in params.items()}

    @field_validator("field_models", mode="before")
    def _validate_field_models(cls, value):
        if value is None:
            return []
        value = [value] if not isinstance(value, list) else value
        if not all(isinstance(i, FieldModel) for i in value):
            raise ValueError("Field models must be FieldModel objects.")
        return value

    @field_validator("parameter_fields", mode="before")
    def validate_parameters(cls, value):
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("Fields must be a dictionary.")
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError("Field names must be strings.")
            if not isinstance(v, FieldInfo):
                raise ValueError("Field values must be FieldInfo objects.")
        return copy(value)

    @field_validator("base_type", mode="before")
    def validate_base(cls, value) -> type[BaseModel]:
        if value is None:
            return BaseModel
        if isinstance(value, type) and issubclass(value, BaseModel):
            return value
        if isinstance(value, BaseModel):
            return value.__class__
        raise ValueError("Base must be a BaseModel subclass or instance.")

    @field_validator("exclude_fields", mode="before")
    def validate_fields(cls, value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, dict):
            value = list(value.keys())
        if isinstance(value, set | tuple):
            value = list(value)
        if isinstance(value, list):
            if not all(isinstance(i, str) for i in value):
                raise ValueError("Field names must be strings.")
            return copy(value)
        raise ValueError("Fields must be a list, set, or dictionary.")

    @field_validator("field_descriptions", mode="before")
    def validate_field_descriptions(cls, value) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("Field descriptions must be a dictionary.")
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError("Field names must be strings.")
            if not isinstance(v, str):
                raise ValueError("Field descriptions must be strings.")
        return value

    @field_validator("inherit_base", mode="before")
    def validate_inherit_base(cls, value) -> bool:
        try:
            return validate_boolean(value)
        except Exception:
            return True

    @field_validator("name", mode="before")
    def validate_name(cls, value) -> str:
        if value is None:
            return "StepModel"
        if not isinstance(value, str):
            raise ValueError("Name must be a string.")
        return value

    @field_validator("field_models", mode="before")
    def validate_field_models(cls, value):
        if value is None:
            return []
        value = [value] if not isinstance(value, list) else value
        if not all(isinstance(i, FieldModel) for i in value):
            raise ValueError("Field models must be FieldModel objects.")
        return value

    @model_validator(mode="after")
    def validate_param_model(self) -> Self:

        if self.base_type is not None:
            self.parameter_fields.update(copy(self.base_type.model_fields))

        self.parameter_fields.update({f.name: f.field_info for f in self.field_models})

        use_keys = list(self.parameter_fields.keys())
        use_keys.extend(list(self._use_keys))

        if self.exclude_fields:
            use_keys = [i for i in use_keys if i not in self.exclude_fields]

        self._use_keys = set(use_keys)

        validators = {}

        for i in self.field_models:
            if i.field_validator is not None:
                validators.update(i.field_validator)
        self._validators = validators

        if self.field_descriptions:
            for i in self.field_models:
                if i.name in self.field_descriptions:
                    i.description = self.field_descriptions[i.name]

        # Prepare class attributes
        class_kwargs = {}
        if self.use_base_kwargs:
            class_kwargs.update(
                {
                    k: getattr(self.base_type, k)
                    for k in self.base_type.__dict__
                    if not k.startswith("__")
                }
            )
        self._class_kwargs = class_kwargs

        if hasattr(self.base_type, "class_name"):
            if callable(self.base_type.class_name):
                self.name = self.base_type.class_name()
            else:
                self.name = self.base_type.class_name
        elif inspect.isclass(self.base_type):
            self.name = self.base_type.__name__

        return self

    def create_new_model(self) -> type[BaseModel]:
        a: type[BaseModel] = create_model(
            self.name,
            __config__=self.config_dict,
            __doc__=self.doc,
            __base__=self.base_type if self.inherit_base else BaseModel,
            __cls_kwargs__=self._class_kwargs,
            __validators__=self._validators,
            **self.use_fields,
        )
        if self.frozen:
            a.model_config.frozen = True
        return a


__all__ = [
    "BaseModel",
    "SchemaModel",
    "FieldModel",
    "OperableModel",
    "Note",
    "NewModelParams",
    "BaseAutoModel",
]
