#!/bin/bash

echo "🧹 Preparing repository for GitHub submission..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for .env files
echo "📋 Checking for sensitive files..."
env_files=$(find . -name ".env" -o -name ".env.*" | grep -v ".env.example" | grep -v ".env.template" | grep -v node_modules | grep -v .venv)

if [ ! -z "$env_files" ]; then
    echo -e "${YELLOW}⚠️  Found .env files that should not be committed:${NC}"
    echo "$env_files"
    echo ""
    echo "Make sure these are listed in .gitignore!"
else
    echo -e "${GREEN}✅ No unprotected .env files found${NC}"
fi

# Check for large files
echo ""
echo "📦 Checking for large files (>50MB)..."
large_files=$(find . -type f -size +50M | grep -v node_modules | grep -v .venv | grep -v .git)

if [ ! -z "$large_files" ]; then
    echo -e "${YELLOW}⚠️  Found large files:${NC}"
    echo "$large_files"
    echo ""
    echo "Consider if these should be committed or added to .gitignore"
else
    echo -e "${GREEN}✅ No large files found${NC}"
fi

# Check for common sensitive patterns
echo ""
echo "🔍 Checking for potential API keys or secrets..."
suspicious=$(grep -r -E "(api_key|apikey|api-key|secret|password|token)" . --include="*.py" --include="*.js" --include="*.ts" --include="*.json" | grep -v node_modules | grep -v .venv | grep -E "=\s*['\"][^'\"]+['\"]" | head -10)

if [ ! -z "$suspicious" ]; then
    echo -e "${YELLOW}⚠️  Found potential sensitive data:${NC}"
    echo "$suspicious"
    echo ""
    echo "Please review these lines carefully!"
else
    echo -e "${GREEN}✅ No obvious secrets found${NC}"
fi

# Check git status
echo ""
echo "📊 Git status:"
git status --short

echo ""
echo "🎯 Recommended next steps:"
echo "1. Review any warnings above"
echo "2. Make sure all .env files have corresponding .env.example files"
echo "3. Run: git add ."
echo "4. Run: git commit -m 'Initial commit: AIM Certificate Challenge submission'"
echo "5. Create a new GitHub repository"
echo "6. Run: git remote add origin [your-repo-url]"
echo "7. Run: git push -u origin main"

echo ""
echo -e "${GREEN}✨ Good luck with your submission!${NC}"