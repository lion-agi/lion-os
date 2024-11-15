import asyncio
import json
import os

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.environ.get("LION_API_KEY")
if not API_KEY:
    raise ValueError("LION_API_KEY not set in environment")

BASE_URL = "http://127.0.0.1:8000"


async def test_api():
    """Test all API endpoints and verify database operations."""
    headers = {"X-API-Key": API_KEY}

    # Configure client with longer timeout and better error handling
    timeout = httpx.Timeout(100.0, connect=30.0)  # 100 seconds timeout
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=timeout) as client:
        try:
            # 1. Test health check
            print("\n1. Testing health check...")
            response = await client.get("/health")
            assert (
                response.status_code == 200
            ), f"Health check failed with status {response.status_code}"
            health_data = response.json()
            print(f"Health check response: {health_data}")
            assert "status" in health_data, "Health check response missing status"
            assert health_data["status"] == "healthy", "Health check status not healthy"

            # 2. Create new session
            print("\n2. Creating new session...")
            response = await client.post("/sessions", headers=headers)
            assert (
                response.status_code == 200
            ), f"Session creation failed with status {response.status_code}"
            session_data = response.json()
            print(f"Create session response: {session_data}")
            assert "session_id" in session_data, "Session response missing session_id"
            session_id = session_data["session_id"]

            # 3. Test agent run with brainstorming
            print("\n3. Testing agent run (brainstorm)...")
            run_data = {
                "session_id": session_id,
                "instruction": {
                    "instruction": "Generate app ideas",
                    "guidance": "Generate ideas for a new mobile app focusing on productivity tools",
                    "context": {
                        "domain": "mobile apps",
                        "focus": "productivity",
                        "target_audience": "professionals",
                    },
                },
            }
            try:
                response = await client.post(
                    "/agent/brainstorm", headers=headers, json=run_data
                )
                print("\nRequest data:", json.dumps(run_data, indent=2))
                print("\nResponse status:", response.status_code)
                print("\nResponse headers:", dict(response.headers))

                response_data = None
                try:
                    response_data = response.json()
                    print("\nResponse data:", json.dumps(response_data, indent=2))
                except json.JSONDecodeError as e:
                    print("\nFailed to decode response as JSON:", str(e))
                    print("\nRaw response text:", response.text)

                assert (
                    response.status_code == 200
                ), f"Brainstorm operation failed with status {response.status_code}"
                if response_data and "detail" in response_data:
                    assert (
                        False
                    ), f"Brainstorm operation failed: {response_data['detail']}"

            except Exception as e:
                print("\nDetailed error information:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                if hasattr(e, "__traceback__"):
                    import traceback

                    print("\nTraceback:")
                    traceback.print_tb(e.__traceback__)
                raise

            # 4. Get session history
            print("\n4. Getting session history...")
            response = await client.get(
                f"/sessions/{session_id}/history", headers=headers
            )
            assert (
                response.status_code == 200
            ), f"Session history retrieval failed with status {response.status_code}"
            history_data = response.json()
            print(f"Session history response: {json.dumps(history_data, indent=2)}")

            # 5. Test selection operation
            print("\n5. Testing selection operation...")
            select_data = {
                "session_id": session_id,
                "instruction": {
                    "instruction": "Select best app idea",
                    "guidance": "Choose the most innovative and practical app idea",
                    "context": {
                        "criteria": ["innovation", "practicality", "market potential"]
                    },
                },
                "choices": ["Task Management App", "Time Tracking Tool", "Focus Timer"],
            }
            response = await client.post(
                "/agent/select", headers=headers, json=select_data
            )
            assert (
                response.status_code == 200
            ), f"Selection operation failed with status {response.status_code}"
            select_result = response.json()
            print(f"Selection response: {json.dumps(select_result, indent=2)}")

            # 6. Delete session
            print("\n6. Deleting session...")
            response = await client.delete(f"/sessions/{session_id}", headers=headers)
            assert (
                response.status_code == 200
            ), f"Session deletion failed with status {response.status_code}"
            delete_result = response.json()
            print(f"Delete session response: {json.dumps(delete_result, indent=2)}")

            print("\n✅ All tests passed successfully!")

        except httpx.HTTPError as e:
            print(f"\n❌ HTTP Error occurred:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            if isinstance(e, httpx.HTTPStatusError):
                print(f"\nResponse status code: {e.response.status_code}")
                try:
                    print(f"Response JSON: {e.response.json()}")
                except json.JSONDecodeError:
                    print(f"Raw response text: {e.response.text}")
            raise
        except AssertionError as e:
            print(f"\n❌ Assertion failed: {str(e)}")
            raise
        except Exception as e:
            print(f"\n❌ Unexpected error:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            if hasattr(e, "__traceback__"):
                import traceback

                print("\nTraceback:")
                traceback.print_tb(e.__traceback__)
            raise


def main():
    """Run the API tests."""
    print("\nStarting API tests...")
    print(f"Using API key: {API_KEY}")
    print(f"Server URL: {BASE_URL}")

    asyncio.run(test_api())


if __name__ == "__main__":
    main()
