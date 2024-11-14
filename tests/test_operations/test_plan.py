from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.operations.plan import plan
from lion.operations.plan.prompt import PROMPT
from lion.protocols.operatives.instruct import INSTRUCT_MODEL_FIELD, InstructModel


@pytest.fixture
def mock_instruct_model():
    instruct = MagicMock(spec=InstructModel)
    instruct.guidance = "Test guidance"
    instruct.model_dump = MagicMock(return_value={"guidance": "Test guidance"})
    instruct.clean_dump = MagicMock(return_value={"guidance": "Test guidance"})
    return instruct


@pytest.mark.asyncio
async def test_run_step_basic(mock_instruct_model):
    """Test basic step execution"""
    mock_response = MagicMock()

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response

        result = await plan.run_instruction(
            mock_instruct_model, MagicMock(spec=Session), Branch(), True, verbose=True
        )

        assert result == mock_response
        mock_operate.assert_called_once()


@pytest.mark.asyncio
async def test_plan_basic(mock_instruct_model):
    """Test basic plan functionality without auto_run"""
    mock_response = MagicMock()

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response

        result = await plan(mock_instruct_model, num_instruct=3, auto_run=False)

        assert result == mock_response
        mock_operate.assert_called_once()
        called_kwargs = mock_operate.call_args.kwargs
        assert "guidance" in called_kwargs
        assert PROMPT.format(num_instruct=3) in called_kwargs["guidance"]


@pytest.mark.asyncio
async def test_plan_with_auto_run(mock_instruct_model):
    """Test plan with auto_run and multiple steps"""
    # Create mock response with steps
    mock_step = MagicMock(spec=InstructModel)
    mock_step.guidance = "Step guidance"
    mock_step.model_dump = MagicMock(return_value={"guidance": "Step guidance"})

    mock_response = MagicMock()
    mock_response.instruct_models = [mock_step, mock_step]  # Two steps

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.side_effect = [mock_response] + [MagicMock()] * len(
            mock_response.instruct_models
        )

        result = await plan(mock_instruct_model, num_instruct=2, auto_run=True)

        assert isinstance(result, list)
        assert len(result) == 3  # Initial result + 2 steps
        assert mock_operate.call_count == 3  # Initial call + 2 steps


@pytest.mark.asyncio
async def test_plan_with_dict_input():
    """Test plan with dictionary input"""
    mock_response = MagicMock()
    instruct_dict = {"guidance": "Test guidance"}

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response

        result = await plan(instruct_dict, auto_run=False)

        assert result == mock_response
        mock_operate.assert_called_once()


@pytest.mark.asyncio
async def test_plan_with_invalid_input():
    """Test plan with invalid input type"""
    with pytest.raises(ValueError):
        await plan("invalid_input")


@pytest.mark.asyncio
async def test_plan_with_return_session(mock_instruct_model):
    """Test plan with return_session option"""
    mock_response = MagicMock()

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response

        result, session = await plan(
            mock_instruct_model, auto_run=False, return_session=True
        )

        assert result == mock_response
        assert isinstance(session, Session)
        mock_operate.assert_called_once()


@pytest.mark.asyncio
async def test_plan_with_verbose(mock_instruct_model):
    """Test plan with verbose output"""
    mock_response = MagicMock()
    mock_response.instruct_models = []

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response

        result = await plan(mock_instruct_model, verbose=True, auto_run=True)

        assert isinstance(result, list)
        mock_operate.assert_called_once()


@pytest.mark.asyncio
async def test_plan_with_branch_kwargs(mock_instruct_model):
    """Test plan with branch_kwargs"""
    mock_response = MagicMock()
    mock_branch = MagicMock(spec=Branch)
    mock_branch.operate = AsyncMock(return_value=mock_response)

    with patch("lion.core.session.session.Session.new_branch") as mock_new_branch:
        mock_new_branch.return_value = mock_branch

        result = await plan(
            mock_instruct_model, branch_kwargs={"name": "test_branch"}, auto_run=False
        )

        assert result == mock_response
        mock_new_branch.assert_called_once()
        mock_branch.operate.assert_called_once()
