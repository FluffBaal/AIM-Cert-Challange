#!/bin/bash
set -e

echo "🚀 Setting up Freelancer Negotiation Helper development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed"

# Install backend dependencies
echo "📚 Installing backend dependencies..."

for service in api ingestion evaluation; do
    echo "  → Installing dependencies for $service..."
    cd backend/$service
    uv pip sync
    cd ../..
done

# Install frontend dependencies
echo "🎨 Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create necessary directories
echo "📁 Creating required directories..."
mkdir -p data data/evaluation_results

# Create .env template if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env template..."
    cat > .env << EOF
# System API Keys (for ingestion and evaluation)
OPENAI_API_KEY=your_openai_key_here
LANGSMITH_API_KEY=your_langsmith_key_here  # Optional

# User API keys are provided through the frontend UI
EOF
fi

echo "✨ Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your API keys to the .env file"
echo "2. Place PDF documents in the data/ directory"
echo "3. Run 'make docker-up' to start all services"
echo "4. Or use 'make dev-api' and 'make dev-frontend' for local development"
echo ""
echo "Happy coding! 🎉"