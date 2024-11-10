import os

import litellm
from dotenv import load_dotenv

litellm.drop_params = True
load_dotenv()


RESERVED_PARAMS = [
    "invoke_action",
    "instruction",
    "clear_messages",
]


class iModel:

    def __init__(self, **kwargs):
        if "api_key" in kwargs:
            try:
                api_key = os.getenv(kwargs["api_key"], None)
                if api_key:
                    kwargs["api_key"] = api_key
            except Exception:
                pass
        self.kwargs = kwargs
        self.acompletion = litellm.acompletion

    async def invoke(self, **kwargs):
        config = {**self.kwargs, **kwargs}
        for i in RESERVED_PARAMS:
            config.pop(i, None)

        return await self.acompletion(**config)
