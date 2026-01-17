#!/bin/bash
# GitHub Push Script for AutoML Tabular
# Run this to push your project to GitHub

set -e  # Exit on error

echo "🚀 Pushing AutoML Tabular to GitHub..."
echo ""

# Initialize git if needed
if [ ! -d .git ]; then
    echo "📦 Initializing git repository..."
    git init
    git branch -M main
fi

# Add remote
echo "🔗 Adding remote origin..."
git remote add origin https://github.com/Mounusha25/automl-tabular.git 2>/dev/null || \
git remote set-url origin https://github.com/Mounusha25/automl-tabular.git

# Show what will be committed
echo ""
echo "📋 Files to be committed:"
git add .
git status --short

echo ""
echo "⚠️  Verify that 'output/' is NOT in the list above (should be gitignored)"
echo ""
read -p "Continue with commit? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Commit
    echo "💾 Committing changes..."
    git commit -m "Initial commit: Production-ready AutoML engine

Features:
- Binary, multiclass, and regression support
- Generic label encoding for all classification tasks
- Tolerance-based model selection with simplicity tie-breaking
- Professional HTML reports with explainability
- Validated on 4 datasets: Titanic, Adult Income, CA Housing, Wine Quality

Includes:
- Complete source code with modular architecture
- 4 sample HTML reports showcasing different problem types
- Example datasets (Titanic, Adult Income, CA Housing, Wine Quality)
- Comprehensive README with usage examples
- Full requirements.txt for easy setup"

    # Push
    echo "⬆️  Pushing to GitHub..."
    git push -u origin main

    echo ""
    echo "✅ SUCCESS! Your project is now live at:"
    echo "   https://github.com/Mounusha25/automl-tabular"
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Go to GitHub and verify everything looks good"
    echo "   2. Add topics: automl, machine-learning, tabular-data, explainability, python"
    echo "   3. Pin this repo on your GitHub profile"
    echo "   4. Add to your resume!"
else
    echo "❌ Aborted. No changes pushed."
fi
