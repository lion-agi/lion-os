from lion.core.models import SchemaModel


class LionIDConfig(SchemaModel):
    n: int
    random_hyphen: bool
    num_hyphens: int
    hyphen_start_index: int
    hyphen_end_index: int
    prefix: str = "ln"
    postfix: str = ""
