# Lion OS - Agent System

A robust agent system with persistent sessions, operation tracking, and a RESTful API.

## Architecture

### Components

1. **Agent**
   - Orchestrates operations (brainstorm, plan, select)
   - Maintains state through sessions
   - Tracks operation history
   - Handles error recovery

2. **Operations**
   - Modular functions that perform specific tasks
   - Current operations:
     - `brainstorm`: Generate ideas and initial thoughts
     - `plan`: Create structured plans from instructions
     - `select`: Make selections from given choices

3. **SessionManager**
   - Handles persistent storage of sessions
   - Manages session lifecycle and expiration
   - Provides caching for better performance
   - Tracks operation history

4. **API Layer**
   - FastAPI endpoints exposing agent functionalities
   - Authentication using API keys
   - Comprehensive error handling
   - Session management endpoints

5. **Memory Storage**
   - SQLite database for session persistence
   - Stores operation history
   - Handles session expiration
   - Thread-safe operations

### Data Flow

1. **API Request**
   - Client sends request with optional session_id
   - Authentication via API key
   - Request validation

2. **Session Management**
   - SessionManager retrieves or creates session
   - Handles session expiration
   - Maintains in-memory cache

3. **Agent Execution**
   - Agent instance uses session for state
   - Executes requested operation
   - Tracks operation history

4. **State Updates**
   - Session state updated after operations
   - Operation history recorded
   - State persisted to database

5. **API Response**
   - Results returned to client
   - Session ID included for continuity
   - Error handling when needed

## Usage

### API Endpoints

```bash
# Create new session
curl -X POST "http://localhost:8000/sessions" \
     -H "X-API-Key: your-secret-key"

# Run agent workflow
curl -X POST "http://localhost:8000/agent/run" \
     -H "X-API-Key: your-secret-key" \
     -H "Content-Type: application/json" \
     -d '{
           "instruction": {"task": "your task here"},
           "session_id": "optional-session-id"
         }'

# Execute brainstorm operation
curl -X POST "http://localhost:8000/agent/brainstorm" \
     -H "X-API-Key: your-secret-key" \
     -H "Content-Type: application/json" \
     -d '{
           "instruction": {"topic": "your topic here"},
           "session_id": "optional-session-id"
         }'

# Execute plan operation
curl -X POST "http://localhost:8000/agent/plan" \
     -H "X-API-Key: your-secret-key" \
     -H "Content-Type: application/json" \
     -d '{
           "instruction": {"goal": "your goal here"},
           "session_id": "optional-session-id"
         }'

# Execute select operation
curl -X POST "http://localhost:8000/agent/select" \
     -H "X-API-Key: your-secret-key" \
     -H "Content-Type: application/json" \
     -d '{
           "instruction": {"criteria": "your criteria here"},
           "choices": ["option1", "option2", "option3"],
           "session_id": "optional-session-id"
         }'

# Get session history
curl -X GET "http://localhost:8000/sessions/{session_id}/history" \
     -H "X-API-Key: your-secret-key"

# Delete session
curl -X DELETE "http://localhost:8000/sessions/{session_id}" \
     -H "X-API-Key: your-secret-key"
```

### Python Client Example

```python
import httpx
import asyncio

async def agent_example():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Create session
        headers = {"X-API-Key": "your-secret-key"}
        session_response = await client.post("/sessions", headers=headers)
        session_id = session_response.json()["session_id"]

        # Run agent workflow
        run_response = await client.post(
            "/agent/run",
            headers=headers,
            json={
                "session_id": session_id,
                "instruction": {"task": "Plan a web application"}
            }
        )

        # Get operation history
        history_response = await client.get(
            f"/sessions/{session_id}/history",
            headers=headers
        )

        print("Operation History:", history_response.json())

# Run example
asyncio.run(agent_example())
```

## Features

- **Persistent Sessions**: Sessions are stored in SQLite database with automatic expiration
- **Operation History**: Track all agent operations with timestamps
- **Authentication**: API key-based security
- **Error Handling**: Comprehensive error handling and logging
- **Async Support**: All operations are asynchronous for better performance
- **Caching**: In-memory caching of active sessions
- **Thread Safety**: Thread-safe database operations

## Future Considerations

1. **Scalability**
   - Implement distributed caching (Redis)
   - Add database sharding support
   - Consider message queue for operations

2. **Security**
   - Add role-based access control
   - Implement rate limiting
   - Add request validation middleware

3. **Monitoring**
   - Add prometheus metrics
   - Implement logging aggregation
   - Add performance tracking

4. **Features**
   - Add websocket support for real-time updates
   - Implement batch operations
   - Add support for custom operations

## Development

### Requirements

- Python 3.8+
- FastAPI
- SQLite3
- Additional dependencies in pyproject.toml

### Setup

1. Clone the repository
2. Install dependencies: `pip install -e .`
3. Run the API: `uvicorn lion.api.agent_api:app --reload`
4. Access API docs: http://localhost:8000/docs

### Testing

Run tests with pytest:

```bash
pytest tests/
```

## License

MIT License - see LICENSE file for details
