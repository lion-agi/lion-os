from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.core.types import ID
from lion.libs.func import alcall
from lion.protocols.operatives.instruct import INSTRUCT_MODEL_FIELD, InstructModel

from .prompt import PROMPT


async def run_instruct(
    ins: InstructModel,
    session: Session,
    branch: Branch,
    auto_run: bool,
    **kwargs,
):

    async def run(ins_):
        b_ = session.split(branch)
        return await run_instruct(ins_, session, b_, False, **kwargs)

    config = {**ins.model_dump(), **kwargs}
    res = await branch.operate(**config)
    branch.msgs.logger.dump()
    instructs = []

    if hasattr(res, "instruct_models"):
        instructs = res.instruct_models

    if auto_run is True and instructs:
        ress = await alcall(instructs, run)
        response_ = []
        for res in ress:
            if isinstance(res, list):
                response_.extend(res)
            else:
                response_.append(res)
        response_.insert(0, res)
        return response_

    return res


async def brainstorm(
    instruct: InstructModel | dict,
    num_instruct: int = 3,
    session: Session | None = None,
    branch: Branch | ID.Ref | None = None,
    auto_run=True,
    branch_kwargs: dict = {},
    **kwargs,
):
    field_models: list = kwargs.get("field_models", [])
    if INSTRUCT_MODEL_FIELD not in field_models:
        field_models.append(INSTRUCT_MODEL_FIELD)

    kwargs["field_models"] = field_models

    if session is not None:
        if branch is not None:
            branch: Branch = session.branches[branch]
        else:
            branch = session.new_branch(**branch_kwargs)
    else:
        session = Session()
        if isinstance(branch, Branch):
            session.branches.include(branch)
            session.default_branch = branch
        if branch is None:
            branch = session.new_branch(**branch_kwargs)

    if isinstance(instruct, InstructModel):
        instruct = instruct.clean_dump()
    if not isinstance(instruct, dict):
        raise ValueError(
            "instruct needs to be an InstructModel obj or a dictionary of valid parameters"
        )

    guidance = instruct.get("guidance", "")
    instruct["guidance"] = f"\n{PROMPT.format(num_instruct=num_instruct)}" + guidance

    res1 = await branch.operate(**instruct, **kwargs)

    instructs = None

    async def run(ins_):
        b_ = session.split(branch)
        return await run_instruct(ins_, session, b_, auto_run, **kwargs)

    if not auto_run:
        return res1

    async with session.branches:
        if hasattr(res1, "instruct_models"):
            instructs: list[InstructModel] = res1.instruct_models
            ress = await alcall(instructs, run)
            response_ = []
            for res in ress:
                if isinstance(res, list):
                    response_.extend(res)
                else:
                    response_.append(res)

            if response_:
                return response_

    return res1
