from lion.protocols.adapters.adapter import AdapterRegistry
from lion.protocols.adapters.json_adapter import JsonAdapter, JsonFileAdapter
from lion.protocols.adapters.pandas_adapter import PandasSeriesAdapter

ADAPTERS = [
    JsonAdapter,
    JsonFileAdapter,
    PandasSeriesAdapter,
]


class ComponentAdapterRegistry(AdapterRegistry):
    _adapters = {k.obj_key: k() for k in ADAPTERS}


__all__ = ["ComponentAdapterRegistry"]
