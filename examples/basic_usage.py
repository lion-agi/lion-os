"""Example script demonstrating basic usage of the Lion agent system."""

import asyncio
import json

from lion.core.session.session_manager import SessionManager
from lion.operator.agent import Agent


async def main():
    # Initialize session manager
    session_manager = SessionManager()

    try:
        # Create a new session
        session_id = session_manager.create_session()
        print(f"\nCreated session: {session_id}")

        # Create a new agent with the session
        agent = Agent(session_id=session_id, session_manager=session_manager)

        # Save some test operations
        session_manager.save_operation_history(
            session_id,
            "test_operation",
            {"task": "test task"},
            {"result": "test result"},
        )

        # Get operation history
        history = session_manager.get_operation_history(session_id)
        print("\nOperation History:")
        print(json.dumps(history, indent=2))

    finally:
        # Cleanup
        session_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
