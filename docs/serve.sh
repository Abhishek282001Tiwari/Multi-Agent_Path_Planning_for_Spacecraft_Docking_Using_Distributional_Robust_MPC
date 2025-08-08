#!/bin/bash
# serve.sh - Local development server for Jekyll site

echo "Starting Jekyll development server..."

# Navigate to docs directory
cd "$(dirname "$0")"

# Install dependencies if needed
if [ ! -d "_site" ]; then
    echo "Installing dependencies..."
    bundle install
fi

# Start Jekyll with live reload
echo "Starting server at http://localhost:4000"
echo "Press Ctrl+C to stop"
bundle exec jekyll serve --livereload --host 0.0.0.0 --port 4000