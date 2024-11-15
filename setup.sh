#!/bin/bash

# Install dependencies
echo "Installing dependencies..."
poetry install

# Create database directory if it doesn't exist
mkdir -p data

# Set up environment variables if .env doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "LION_API_KEY=dev-key-123456789" > .env
    echo "LION_ENV=development" >> .env
    echo "DATABASE_URL=sqlite:///data/sessions.db" >> .env
fi

echo "Setup complete! You can now:"
echo "1. Start the server: python server.py"
echo "2. Run tests: python test_api.py"
