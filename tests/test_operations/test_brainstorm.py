from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.operations.brainstorm import brainstorm
from lion.operations.brainstorm.prompt import PROMPT
from lion.protocols.operatives.instruct import INSTRUCT_MODEL_FIELD, InstructModel


@pytest.fixture
def mock_instruct_model():
    instruct = MagicMock(spec=InstructModel)
    instruct.guidance = "Test guidance"
    instruct.model_dump = MagicMock(return_value={"guidance": "Test guidance"})
    instruct.clean_dump = MagicMock(return_value={"guidance": "Test guidance"})
    return instruct


@pytest.mark.asyncio
async def test_brainstorm_basic(mock_instruct_model):
    """Test basic brainstorm functionality with mocked API response"""
    mock_response = MagicMock()
    mock_response.instruct_models = []

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response

        result = await brainstorm(mock_instruct_model, num_instruct=3, auto_run=False)

        assert result == mock_response
        mock_operate.assert_called_once()


@pytest.mark.asyncio
async def test_brainstorm_with_auto_run(mock_instruct_model):
    """Test brainstorm with auto_run and nested instructions"""
    # Create a mock response with nested instructions
    nested_response = MagicMock()
    nested_instruct = MagicMock(spec=InstructModel)
    nested_instruct.guidance = "Nested guidance"
    nested_instruct.model_dump = MagicMock(return_value={"guidance": "Nested guidance"})
    nested_response.instruct_models = [nested_instruct]

    # Create mock for nested instruction result
    nested_result = MagicMock()
    nested_result.instruct_models = []

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        # First call returns response with nested instructions, subsequent calls return empty result
        mock_operate.side_effect = [nested_response, nested_result]

        result = await brainstorm(mock_instruct_model, num_instruct=3, auto_run=True)

        assert isinstance(result, list)
        assert len(result) > 0
        assert (
            mock_operate.call_count >= 2
        )  # Should be called for initial and nested instructions


@pytest.mark.asyncio
async def test_brainstorm_with_dict_input():
    """Test brainstorm with dictionary input instead of InstructModel"""
    instruct_dict = {"guidance": "Test guidance"}
    mock_response = MagicMock()
    mock_response.instruct_models = []

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response

        result = await brainstorm(instruct_dict, num_instruct=3, auto_run=False)

        assert result == mock_response
        mock_operate.assert_called_once()


@pytest.mark.asyncio
async def test_brainstorm_with_invalid_input():
    """Test brainstorm with invalid input type"""
    with pytest.raises(ValueError):
        await brainstorm("invalid_input")


@pytest.mark.asyncio
async def test_brainstorm_with_return_session():
    """Test brainstorm with return_session option"""
    instruct_dict = {"guidance": "Test guidance"}
    mock_response = MagicMock()
    mock_response.instruct_models = []

    # Create a mock session and branch
    mock_session = MagicMock(spec=Session)
    mock_branch = MagicMock(spec=Branch)

    # Mock both Branch.operate and Session creation
    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response
        mock_branch.operate = mock_operate

        # Mock Session creation
        with patch("lion.core.session.session.Session", return_value=mock_session):
            # Mock new_branch method
            mock_session.new_branch.return_value = mock_branch

            # Execute brainstorm with return_session=True
            result = await brainstorm(
                instruct_dict,
                session=mock_session,
                num_instruct=3,
                auto_run=False,
                return_session=True,
            )

            # Verify the result is a tuple containing the response and session
            assert isinstance(result, tuple), "Result should be a tuple"
            assert len(result) == 2, "Result tuple should have length 2"
            response, session = result
            assert response == mock_response, "First element should be the API response"
            assert session == mock_session, "Second element should be the session"


@pytest.mark.asyncio
async def test_brainstorm_with_verbose(mock_instruct_model):
    """Test brainstorm with verbose output"""
    mock_response = MagicMock()
    mock_response.instruct_models = []

    with patch(
        "lion.core.session.branch.Branch.operate", new_callable=AsyncMock
    ) as mock_operate:
        mock_operate.return_value = mock_response

        result = await brainstorm(
            mock_instruct_model, num_instruct=3, auto_run=False, verbose=True
        )

        assert result == mock_response
        mock_operate.assert_called_once()
