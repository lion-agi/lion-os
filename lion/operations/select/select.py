from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from lion import Branch
from lion.protocols.operatives.instruct import InstructModel

from .prompt import PROMPT
from .utils import parse_selection, parse_to_representation


class SelectionModel(BaseModel):
    selected: list = Field(default_factory=list)


async def select(
    instruct: dict | InstructModel,
    choices: list[str] | type[Enum] | dict[str, Any],
    max_num_selections: int = 1,
    branch: Branch = None,
    branch_kwargs: dict = {},
    return_branch=False,
    **kwargs,
) -> SelectionModel | tuple[SelectionModel, Branch]:

    branch = branch or Branch(**branch_kwargs)
    selections, contents = parse_to_representation(choices)
    prompt = PROMPT.format(max_num_selections=max_num_selections, choices=selections)

    if isinstance(instruct, InstructModel):
        instruct = instruct.clean_dump()

    instruct = instruct or {}

    if instruct.get("instruction", None) is not None:
        instruct["instruction"] = f"{instruct['instruction']}\n\n{prompt} \n\n "
    else:
        instruct["instruction"] = prompt

    context = instruct.get("context", None) or []
    context = [context] if not isinstance(context, list) else context
    context.extend([{k: v} for k, v in zip(selections, contents)])
    instruct["context"] = context

    response_model: SelectionModel = await branch.operate(
        operative_model=SelectionModel,
        **kwargs,
        **instruct,
    )
    selected = response_model
    if isinstance(response_model, BaseModel) and hasattr(response_model, "selected"):
        selected = response_model.selected
    selected = [selected] if not isinstance(selected, list) else selected

    corrected_selections = [parse_selection(i, choices) for i in selected]

    if isinstance(response_model, BaseModel):
        response_model.selected = corrected_selections

    elif isinstance(response_model, dict):
        response_model["selected"] = corrected_selections

    if return_branch:
        return response_model, branch
    return response_model
