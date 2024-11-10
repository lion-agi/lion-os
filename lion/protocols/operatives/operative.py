from typing import Self

from pydantic import BaseModel, Field, PrivateAttr, model_validator
from pydantic.fields import FieldInfo

from lion.core.models import FieldModel, NewModelParams, OperableModel
from lion.libs.parse import UNDEFINED, to_json, validate_keys


class Operative(OperableModel):

    name: str | None = None

    request_params: NewModelParams | None = Field(default=None)
    request_type: type[BaseModel] | None = Field(default=None)

    response_params: NewModelParams | None = Field(default=None)
    response_type: type[BaseModel] | None = Field(default=None)
    response_model: OperableModel | None = Field(default=None)
    response_str_dict: dict | str | None = Field(default=None)

    auto_retry_parse: bool = True
    max_retries: int = 3
    _should_retry: bool = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.request_type is None:
            self.request_type = self.request_params.create_new_model()
        if self.name is None:
            self.name = self.request_params.name or self.request_type.__name__
        return self

    def raise_validate_pydantic(self, text: str):
        d_ = to_json(text, fuzzy_parse=True)
        if isinstance(d_, list | tuple) and len(d_) == 1:
            d_ = d_[0]
        try:
            d_ = validate_keys(
                d_, self.request_type.model_fields, handle_unmatched="raise"
            )
            d_ = {k: v for k, v in d_.items() if v != UNDEFINED}
            self.response_model = self.request_type.model_validate(d_)
            self._should_retry = False
        except Exception:
            self.response_str_dict = d_
            self._should_retry = True

    def force_validate_pydantic(self, text: str):
        d_ = text
        try:
            d_ = to_json(text, fuzzy_parse=True)
            if isinstance(d_, list | tuple) and len(d_) == 1:
                d_ = d_[0]
            d_ = validate_keys(
                d_, self.request_type.model_fields, handle_unmatched="force"
            )
            d_ = {k: v for k, v in d_.items() if v != UNDEFINED}
            self.response_model = self.request_type.model_validate(d_)
            self._should_retry = False
        except Exception:
            self.response_str_dict = d_
            self.response_model = None
            self._should_retry = True

    def update_response_model(
        self, text: str = None, data: dict = None
    ) -> BaseModel | dict | str | None:

        if text is None and data is None:
            raise ValueError("Either text or data must be provided.")

        if text:
            self.response_str_dict = text
            try:
                self.raise_validate_pydantic(text)
            except Exception:
                self.force_validate_pydantic(text)

        if data and self.response_type:
            d_ = self.response_model.model_dump()
            d_.update(data)
            self.response_model = self.response_type.model_validate(d_)

        if not self.response_model and isinstance(self.response_str_dict, list):
            try:
                self.response_model = [
                    self.request_type.model_validate(d_)
                    for d_ in self.response_str_dict
                ]
            except Exception:
                pass

        return self.response_model or self.response_str_dict

    def create_response_type(
        self,
        response_params: NewModelParams = None,
        field_models: list[FieldModel] = [],
        parameter_fields: dict[str, FieldInfo] = None,
        exclude_fields: list = [],
        field_descriptions: dict = {},
        inherit_base: bool = True,
        use_base_kwargs: bool = False,
        config_dict: dict | None = None,
        doc: str | None = None,
        frozen: bool = False,
        validators=None,
    ):
        self.response_params = response_params or NewModelParams(
            parameter_fields=parameter_fields,
            field_models=field_models,
            exclude_fields=exclude_fields,
            field_descriptions=field_descriptions,
            inherit_base=inherit_base,
            use_base_kwargs=use_base_kwargs,
            config_dict=config_dict,
            doc=doc,
            frozen=frozen,
            base_type=self.request_params.base_type,
        )
        if validators and isinstance(validators, dict):
            self.response_params._validators.update(validators)

        self.response_type = self.response_params.create_new_model()
