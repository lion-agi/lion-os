from lion.core.typing import ID, UNDEFINED, Any, BaseModel, IDError, Literal, LnID, Note
from lion.integrations.pydantic_ import break_down_pydantic_annotation
from lion.libs.utils import time

DEFAULT_SYSTEM = "You are a helpful AI assistant. Let's think step by step."


def format_system_content(
    system_datetime: bool | str | None,
    system_message: str,
) -> Note:
    """Format the system content with optional datetime information."""
    content = Note(system=str(system_message or DEFAULT_SYSTEM))
    if system_datetime:
        if isinstance(system_datetime, str):
            content["system_datetime"] = system_datetime
        else:
            content["system_datetime"] = time(type_="iso", timespec="minutes")
    return content


def prepare_request_response_format(request_fields: dict) -> str:
    return (
        "**MUST RETURN JSON-PARSEABLE RESPONSE ENCLOSED BY JSON CODE BLO"
        f"CKS.** \n```json\n{request_fields}\n```"
    ).strip()


def format_image_item(idx: str, x: str, /) -> dict[str, Any]:
    """Create an image_url dict for content formatting."""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{idx}",
            "detail": x,
        },
    }


def format_text_item(item):
    msg = ""
    item = [item] if not isinstance(item, list) else item
    for j in item:
        if isinstance(j, dict):
            for k, v in j.items():
                msg += f"- {k}: {v} \n\n"
        else:
            msg += f"{j}\n"
    return msg


def format_text_content(content: dict):
    if "plain_content" in content and isinstance(content["plain_content"], str):
        return content["plain_content"]

    msg = "\n---\n # Task\n"
    for k, v in content.items():
        if k in [
            "guidance",
            "instruction",
            "context",
            "request_response_format",
            "tool_schemas",
        ]:
            if k == "request_response_format":
                k = "response format"
            msg += f"## **Task {k}**\n{format_text_item(v)}\n\n"
    msg += "\n\n---\n"
    return msg


def format_image_content(
    text_content: str,
    images: list,
    image_detail: Literal["low", "high", "auto"],
) -> dict[str, Any]:
    """Format text content with images for message content."""
    content = [{"type": "text", "text": text_content}]
    content.extend(format_image_item(i, image_detail) for i in images)
    return content


def prepare_instruction_content(
    guidance: str | None = None,
    instruction: str | None = None,
    context: str | dict | list | None = None,
    request_fields: dict | list[str] | None = None,
    plain_content: str | None = None,
    request_model: BaseModel = None,
    images: str | list | None = None,
    image_detail: Literal["low", "high", "auto"] | None = None,
    tool_schemas: dict | None = None,
) -> Note:
    """Prepare the content for an instruction message."""

    if request_fields and request_model:
        raise ValueError("only one of request_fields or request_model can be provided")

    out_ = {"context": []}
    if guidance:
        out_["guidance"] = guidance
    if instruction:
        out_["instruction"] = instruction
    if context:
        if isinstance(context, list):
            out_["context"].extend(context)
        else:
            out_["context"].append(context)
    if images:
        out_["images"] = images if isinstance(images, list) else [images]
        out_["image_detail"] = image_detail or "low"

    if tool_schemas:
        out_["tool_schemas"] = tool_schemas

    if request_model:
        out_["request_model"] = request_model
        request_fields = break_down_pydantic_annotation(request_model)
        out_["context"].append(
            {"respond_schema_info": request_model.model_json_schema()}
        )

    if request_fields:
        _fields = request_fields if isinstance(request_fields, dict) else {}
        if not isinstance(request_fields, dict):
            _fields = {i: "..." for i in request_fields}
        out_["request_fields"] = _fields
        out_["request_response_format"] = prepare_request_response_format(
            request_fields=_fields
        )

    if plain_content:
        out_["plain_content"] = plain_content

    return Note(
        **{k: v for k, v in out_.items() if v not in [None, UNDEFINED]},
    )


def validate_sender_recipient(
    value: Any, /
) -> LnID | Literal["system", "user", "N/A", "assistant"]:
    """
    Validate the sender and recipient fields for mail-like communication.

    Args:
        value (Any): The value to validate.

    Returns:
        str: The validated sender or recipient value.

    Raises:
        ValueError: If the value is not a valid sender or recipient.
    """
    if value in ["system", "user", "N/A", "assistant"]:
        return value

    if value is None:
        return "N/A"

    try:
        return ID.get_id(value)
    except IDError as e:
        raise ValueError("Invalid sender or recipient") from e
