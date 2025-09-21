#!/usr/bin/env python3
"""
Streamlit Cloud Deployment Helper Script
This script helps you deploy your OMR Evaluation System to Streamlit Cloud
"""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and return the result"""
    print(f"🔧 {description}")
    print(f"   Command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    print(f"✅ Success: {result.stdout.strip()}")
    return True

def check_file_exists(filepath):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {filepath} exists")
        return True
    else:
        print(f"❌ {filepath} not found")
        return False

def main():
    print("🚀 Streamlit Cloud Deployment Helper")
    print("=" * 50)
    print("ScoreVision Pro - OMR Evaluation System")
    print("=" * 50)

    # Check required files
    print("\n📋 Checking required files...")
    required_files = [
        "streamlit_app.py",
        "requirements.txt",
        "packages.txt",
        "streamlit/config.toml",
        ".streamlit/secrets.toml"
    ]

    missing_files = []
    for file in required_files:
        if not check_file_exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"\n❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure all required files are present before deployment.")
        return

    # Check git repository
    print("\n🔍 Checking git repository...")
    if not os.path.exists(".git"):
        print("📁 Initializing git repository...")
        if not run_command("git init", "Initialize git repository"):
            return

    # Check git status
    print("📊 Checking git status...")
    run_command("git status", "Check git status")

    # Add files to git
    print("\n📦 Adding files to git...")
    if not run_command("git add .", "Add all files to git"):
        return

    # Commit files
    print("\n💾 Committing files...")
    commit_message = "Deploy OMR Evaluation System to Streamlit Cloud"
    if not run_command(f'git commit -m "{commit_message}"', "Commit files"):
        return

    # Check if remote origin exists
    print("\n🌐 Checking remote repository...")
    result = subprocess.run("git remote -v", shell=True, capture_output=True, text=True)
    if "origin" not in result.stdout:
        print("❌ No remote repository configured.")
        print("\n📝 Please set up your GitHub repository:")
        print("1. Create a new repository on GitHub")
        print("2. Copy the repository URL")
        print("3. Run: git remote add origin YOUR_REPO_URL")
        print("4. Run: git push -u origin main")

        setup_github = input("\nDo you need help setting up GitHub? (y/n): ").lower()
        if setup_github == 'y':
            webbrowser.open("https://github.com/new")
        return
    else:
        print("✅ Remote repository configured")

    # Push to GitHub
    print("\n⬆️ Pushing to GitHub...")
    if not run_command("git push", "Push to GitHub"):
        return

    # Get repository URL
    result = subprocess.run("git remote get-url origin", shell=True, capture_output=True, text=True)
    repo_url = result.stdout.strip()
    print(f"✅ Code pushed to: {repo_url}")

    # Open Streamlit Cloud
    print("\n🌐 Opening Streamlit Cloud...")
    print("Please follow these steps:")
    print("1. Go to: https://share.streamlit.io")
    print("2. Click 'New app'")
    print("3. Connect your GitHub repository")
    print("4. Set main file path to: streamlit_app.py")
    print("5. Click 'Deploy'")

    open_streamlit = input("\nOpen Streamlit Cloud now? (y/n): ").lower()
    if open_streamlit == 'y':
        webbrowser.open("https://share.streamlit.io")

    print("\n🎉 Deployment preparation completed!")
    print("\n📋 Next steps:")
    print("1. Complete the deployment on Streamlit Cloud")
    print("2. Your app will be available at:")
    print("   https://YOUR_USERNAME-streamlit-app-YOUR_REPO_NAME.streamlit.app")
    print("3. Share the link with your evaluators!")

    print("\n📖 For detailed instructions, see: README_Streamlit_Deployment.md")

if __name__ == "__main__":
    main()
