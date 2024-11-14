from lion.core.generic import Element, Log
from lion.core.typing import Any, Enum, NoReturn, PrivateAttr, override
from lion.settings import Settings, TimedFuncCallConfig


class EventStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ObservableAction(Element):

    status: EventStatus = EventStatus.PENDING
    execution_time: float | None = None
    execution_response: Any = None
    execution_error: str | None = None
    _timed_config: TimedFuncCallConfig | None = PrivateAttr(None)
    _content_fields: list = PrivateAttr(["execution_response"])

    @override
    def __init__(
        self, timed_config: dict | TimedFuncCallConfig | None, **kwargs: Any
    ) -> None:
        super().__init__()
        if timed_config is None:
            self._timed_config = Settings.Config.TIMED_CALL

        else:
            if isinstance(timed_config, TimedFuncCallConfig):
                timed_config = timed_config.to_dict()
            if isinstance(timed_config, dict):
                timed_config = {**timed_config, **kwargs}
            timed_config = TimedFuncCallConfig(**timed_config)
            self._timed_config = timed_config

    def to_log(self) -> Log:
        """
        Convert the action to a log entry. Will forcefully convert all fields
        into a dictionary or json serializable format.

        Returns:
            BaseLog: A log entry representing the action.
        """
        dict_ = self.to_dict()
        dict_["status"] = self.status.value
        content = {k: dict_[k] for k in self._content_fields if k in dict_}
        loginfo = {k: dict_[k] for k in dict_ if k not in self._content_fields}
        return Log(content=content, loginfo=loginfo)

    @classmethod
    def from_dict(cls, data: dict, /, **kwargs: Any) -> NoReturn:
        """Event cannot be re-created."""
        raise NotImplementedError(
            "An event cannot be recreated. Once it's done, it's done."
        )


__all__ = ["ObservableAction"]
# File: autoos/action/base.py
