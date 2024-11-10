from pydantic import BaseModel
from pydantic.fields import FieldInfo

from lion.core.types import FieldModel, NewModelParams
from lion.protocols.operatives.operative import Operative

from .action import (
    ACTION_REQUESTS_FIELD,
    ACTION_REQUIRED_FIELD,
    ACTION_RESPONSES_FIELD,
    ActionRequestModel,
    ActionResponseModel,
)
from .reason import REASON_FIELD, ReasonModel


class StepModel(BaseModel):
    title: str
    description: str
    reason: ReasonModel | None = REASON_FIELD.field_info
    action_requests: list[ActionRequestModel] = ACTION_REQUESTS_FIELD.field_info
    action_required: bool = ACTION_REQUIRED_FIELD.field_info
    action_responses: list[ActionResponseModel] = ACTION_RESPONSES_FIELD.field_info


class Step:

    @staticmethod
    def request_operative(
        *,
        operative_name: str = None,
        reason: bool = False,
        actions: bool = False,
        request_params: NewModelParams = None,
        parameter_fields: dict[str, FieldInfo] = None,
        base_type: type[BaseModel] = None,
        field_models: list[FieldModel] = None,
        exclude_fields: list = None,
        new_model_name: str | None = None,
        field_descriptions: dict = None,
        inherit_base: bool = True,
        use_base_kwargs: bool = False,
        config_dict: dict | None = None,
        doc: str | None = None,
        frozen: bool = False,
    ) -> Operative:

        field_models = field_models or []
        if reason:
            field_models.append(REASON_FIELD)
        if actions:
            field_models.extend(
                [
                    ACTION_REQUESTS_FIELD,
                    ACTION_REQUIRED_FIELD,
                ]
            )
        request_params = request_params or NewModelParams(
            parameter_fields=parameter_fields,
            base_type=base_type,
            field_models=field_models,
            exclude_fields=exclude_fields,
            name=new_model_name,
            field_descriptions=field_descriptions,
            inherit_base=inherit_base,
            use_base_kwargs=use_base_kwargs,
            config_dict=config_dict,
            doc=doc,
            frozen=frozen,
        )
        return Operative(name=operative_name, request_params=request_params)

    @staticmethod
    def respond_operative(
        *,
        operative: Operative,
        additional_data: dict = {},
        response_params: NewModelParams = None,
        field_models: list[FieldModel] = [],
        frozen_reponse: bool = False,
        response_config_dict=None,
        response_doc=None,
        exclude_fields=None,
    ) -> Operative:

        operative = Step._create_response_type(
            operative=operative,
            response_params=response_params,
            field_models=field_models,
            frozen_reponse=frozen_reponse,
            response_config_dict=response_config_dict,
            response_doc=response_doc,
            exclude_fields=exclude_fields,
        )

        data = operative.response_model.model_dump()
        data.update(additional_data or {})
        operative.response_model = operative.response_type.model_validate(data)
        return operative

    @staticmethod
    def _create_response_type(
        operative: Operative,
        response_params: NewModelParams = None,
        response_validators: dict = None,
        frozen_reponse: bool = False,
        response_config_dict=None,
        response_doc=None,
        field_models=None,
        exclude_fields=None,
    ) -> Operative:

        field_models = field_models or []

        if (
            hasattr(operative.request_type, "action_required")
            and operative.response_model.action_required
        ):
            field_models.extend(
                [
                    ACTION_RESPONSES_FIELD,
                    ACTION_REQUIRED_FIELD,
                    ACTION_REQUESTS_FIELD,
                ]
            )
        if hasattr(operative.request_type, "reason"):
            field_models.extend([REASON_FIELD])

        exclude_fields = exclude_fields or []
        exclude_fields.extend(operative.request_params.exclude_fields)

        operative.create_response_type(
            response_params=response_params,
            field_models=field_models,
            exclude_fields=exclude_fields,
            doc=response_doc,
            config_dict=response_config_dict,
            frozen=frozen_reponse,
            validators=response_validators,
        )
        return operative
