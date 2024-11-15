from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lion.core.session.branch import Branch
from lion.operations.select.prompt import PROMPT
from lion.operations.select.select import SelectionModel, select
from lion.protocols.operatives.instruct import InstructModel


class TestChoices(Enum):
    OPTION_A = "Option A"
    OPTION_B = "Option B"
    OPTION_C = "Option C"


@pytest.fixture
def mock_instruct_model():
    instruct = MagicMock(spec=InstructModel)
    instruct.instruction = "Test instruction"
    instruct.clean_dump = MagicMock(return_value={"instruction": "Test instruction"})
    return instruct


@pytest.mark.asyncio
async def test_select_basic():
    """Test basic selection with list choices"""
    choices = ["Option A", "Option B", "Option C"]
    mock_response = SelectionModel(selected=["Option A"])

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response

        result = await select(
            {"instruction": "Select an option"}, choices=choices, max_num_selections=1
        )

        assert isinstance(result, SelectionModel)
        assert result.selected == ["Option A"]
        mock_operate.assert_called_once()
        called_kwargs = mock_operate.call_args.kwargs
        assert (
            PROMPT.format(max_num_selections=1, choices=choices)
            in called_kwargs["instruction"]
        )


@pytest.mark.asyncio
async def test_select_with_enum():
    """Test selection with Enum choices"""
    mock_response = SelectionModel(selected=["Option A"])

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response

        result = await select(
            {"instruction": "Select an option"},
            choices=TestChoices,
            max_num_selections=1,
        )

        assert isinstance(result, SelectionModel)
        assert result.selected == [
            TestChoices.OPTION_A
        ]  # Compare with actual Enum value
        mock_operate.assert_called_once()


@pytest.mark.asyncio
async def test_select_with_dict():
    """Test selection with dictionary choices"""
    choices = {"a": "Option A", "b": "Option B", "c": "Option C"}
    mock_response = SelectionModel(selected=["Option A"])

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response

        result = await select(
            {"instruction": "Select an option"}, choices=choices, max_num_selections=1
        )

        assert isinstance(result, SelectionModel)
        assert result.selected == ["Option A"]
        mock_operate.assert_called_once()


@pytest.mark.asyncio
async def test_select_multiple():
    """Test selecting multiple options"""
    choices = ["Option A", "Option B", "Option C"]
    mock_response = SelectionModel(selected=["Option A", "Option B"])

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response

        result = await select(
            {"instruction": "Select options"}, choices=choices, max_num_selections=2
        )

        assert isinstance(result, SelectionModel)
        assert len(result.selected) == 2
        assert set(result.selected) == {"Option A", "Option B"}
        mock_operate.assert_called_once()


@pytest.mark.asyncio
async def test_select_with_instruct_model(mock_instruct_model):
    """Test selection with InstructModel input"""
    choices = ["Option A", "Option B", "Option C"]
    mock_response = SelectionModel(selected=["Option A"])

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response

        result = await select(
            mock_instruct_model, choices=choices, max_num_selections=1
        )

        assert isinstance(result, SelectionModel)
        assert result.selected == ["Option A"]
        mock_operate.assert_called_once()


@pytest.mark.asyncio
async def test_select_with_return_branch():
    """Test selection with return_branch option"""
    choices = ["Option A", "Option B", "Option C"]
    mock_response = SelectionModel(selected=["Option A"])

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response

        result, branch = await select(
            {"instruction": "Select an option"},
            choices=choices,
            max_num_selections=1,
            return_branch=True,
        )

        assert isinstance(result, SelectionModel)
        assert result.selected == ["Option A"]
        assert isinstance(branch, Branch)
        mock_operate.assert_called_once()


@pytest.mark.asyncio
async def test_select_with_verbose():
    """Test selection with verbose output"""
    choices = ["Option A", "Option B", "Option C"]
    mock_response = SelectionModel(selected=["Option A"])

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response

        result = await select(
            {"instruction": "Select an option"},
            choices=choices,
            max_num_selections=1,
            verbose=True,
        )

        assert isinstance(result, SelectionModel)
        assert result.selected == ["Option A"]
        mock_operate.assert_called_once()


@pytest.mark.asyncio
async def test_select_with_branch_kwargs():
    """Test selection with branch_kwargs"""
    choices = ["Option A", "Option B", "Option C"]
    mock_response = SelectionModel(selected=["Option A"])

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response

        result = await select(
            {"instruction": "Select an option"},
            choices=choices,
            max_num_selections=1,
            branch_kwargs={"name": "test_branch"},
        )

        assert isinstance(result, SelectionModel)
        assert result.selected == ["Option A"]
        mock_operate.assert_called_once()
