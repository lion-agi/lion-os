"""Test script for the Lion Agent API."""

import asyncio
import os

import httpx

# Get API key from environment variable with fallback to development key
API_KEY = os.environ.get("LION_API_KEY", "dev-key-123")
BASE_URL = "http://127.0.0.1:8000"


async def test_api():
    """Test the Lion Agent API endpoints."""
    headers = {"X-API-Key": API_KEY}

    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # Test health check
        print("\nTesting health check...")
        response = await client.get("/health")
        print(f"Health check response: {response.json()}")

        # Create new session
        print("\nCreating new session...")
        try:
            response = await client.post("/sessions", headers=headers)
            if response.status_code == 401:
                print(
                    "Error: Invalid API key. Make sure LION_API_KEY environment variable is set correctly."
                )
                return
            print(f"Create session response: {response.json()}")

            if response.status_code == 200:
                session_id = response.json()["session_id"]

                # Test agent run
                print("\nTesting agent run...")
                run_data = {
                    "session_id": session_id,
                    "instruction": {"task": "Test task"},
                }
                response = await client.post(
                    "/agent/run", headers=headers, json=run_data
                )
                print(f"Agent run response: {response.json()}")

                # Get session history
                print("\nGetting session history...")
                response = await client.get(
                    f"/sessions/{session_id}/history", headers=headers
                )
                print(f"Session history response: {response.json()}")

                # Delete session
                print("\nDeleting session...")
                response = await client.delete(
                    f"/sessions/{session_id}", headers=headers
                )
                print(f"Delete session response: {response.json()}")
        except httpx.HTTPError as e:
            print(f"Error occurred: {str(e)}")
            if isinstance(e, httpx.HTTPStatusError):
                print(f"Response: {e.response.json()}")


def main():
    """Main entry point with setup instructions."""
    if "LION_API_KEY" not in os.environ and not API_KEY.startswith("dev-"):
        print(
            """
Setup Instructions:
------------------
1. Set your API key as an environment variable:

   # For Unix/Linux/MacOS:
   export LION_API_KEY=your-api-key

   # For Windows:
   set LION_API_KEY=your-api-key

2. For development/testing, you can also use a development key that starts with 'dev-'
   (minimum 16 characters required)

3. In production, set LION_ENV=production to disable development keys:
   export LION_ENV=production
"""
        )

    asyncio.run(test_api())


if __name__ == "__main__":
    main()
