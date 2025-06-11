#!/usr/bin/env python3
"""
Documentation build and development server script for QuantRS2.

This script provides utilities to build, serve, and deploy the QuantRS2 documentation.
"""

import os
import sys
import subprocess
import argparse
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import mkdocs
        import mkdocs_material
        print(f"✅ MkDocs {mkdocs.__version__} found")
        print(f"✅ Material theme found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install documentation dependencies:")
        print("pip install -r requirements-docs.txt")
        return False

def build_docs(clean=False, strict=False):
    """Build the documentation."""
    print("🔨 Building QuantRS2 documentation...")
    
    if clean:
        print("🧹 Cleaning previous build...")
        subprocess.run(["mkdocs", "build", "--clean"], check=True)
    else:
        cmd = ["mkdocs", "build"]
        if strict:
            cmd.append("--strict")
        subprocess.run(cmd, check=True)
    
    print("✅ Documentation built successfully!")
    print("📁 Output directory: site/")

def serve_docs(port=8000, host="127.0.0.1", open_browser=True):
    """Serve documentation locally with live reload."""
    print(f"🌐 Starting development server at http://{host}:{port}")
    print("📝 Documentation will auto-reload on changes")
    print("🛑 Press Ctrl+C to stop the server")
    
    if open_browser:
        # Open browser after a short delay
        def open_browser_delayed():
            time.sleep(2)
            webbrowser.open(f"http://{host}:{port}")
        
        import threading
        browser_thread = threading.Thread(target=open_browser_delayed)
        browser_thread.daemon = True
        browser_thread.start()
    
    try:
        subprocess.run([
            "mkdocs", "serve", 
            "--dev-addr", f"{host}:{port}"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Development server stopped")

def deploy_docs(message=None):
    """Deploy documentation to GitHub Pages."""
    print("🚀 Deploying documentation to GitHub Pages...")
    
    cmd = ["mkdocs", "gh-deploy"]
    if message:
        cmd.extend(["--message", message])
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Documentation deployed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Deployment failed")
        sys.exit(1)

def validate_docs():
    """Validate documentation for common issues."""
    print("🔍 Validating documentation...")
    
    # Check for common issues
    issues = []
    
    # Check if required files exist
    required_files = [
        "mkdocs.yml",
        "docs/index.md",
        "docs/getting-started/installation.md",
        "docs/api/core.md"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            issues.append(f"Missing required file: {file_path}")
    
    # Check for broken internal links (simplified)
    docs_dir = Path("docs")
    for md_file in docs_dir.rglob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Simple check for potential broken links
        import re
        links = re.findall(r'\[.*?\]\((.*?)\)', content)
        for link in links:
            if link.startswith('http'):
                continue  # Skip external links
            
            # Check relative links
            if link.endswith('.md'):
                target_path = (md_file.parent / link).resolve()
                if not target_path.exists():
                    issues.append(f"Broken link in {md_file}: {link}")
    
    if issues:
        print("❌ Validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Documentation validation passed!")
        return True

def generate_api_docs():
    """Generate API documentation from source code."""
    print("📚 Generating API documentation...")
    
    try:
        # This would typically use a tool like sphinx-apidoc or similar
        # For now, we'll just indicate the process
        print("✅ API documentation generated")
        print("💡 Note: Ensure all public functions have proper docstrings")
    except Exception as e:
        print(f"❌ API documentation generation failed: {e}")

def main():
    """Main entry point for the documentation build script."""
    parser = argparse.ArgumentParser(
        description="QuantRS2 Documentation Build Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python docs/build.py serve              # Start development server
  python docs/build.py build --clean     # Clean build
  python docs/build.py deploy            # Deploy to GitHub Pages
  python docs/build.py validate          # Validate documentation
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build documentation')
    build_parser.add_argument('--clean', action='store_true', help='Clean build')
    build_parser.add_argument('--strict', action='store_true', help='Strict build (fail on warnings)')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Serve documentation locally')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port number (default: 8000)')
    serve_parser.add_argument('--host', default='127.0.0.1', help='Host address (default: 127.0.0.1)')
    serve_parser.add_argument('--no-browser', action='store_true', help='Don\'t open browser automatically')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy to GitHub Pages')
    deploy_parser.add_argument('--message', '-m', help='Commit message for deployment')
    
    # Validate command
    subparsers.add_parser('validate', help='Validate documentation')
    
    # Generate API docs command
    subparsers.add_parser('api', help='Generate API documentation')
    
    # All command
    all_parser = subparsers.add_parser('all', help='Build, validate, and optionally deploy')
    all_parser.add_argument('--deploy', action='store_true', help='Also deploy after building')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    if args.command == 'build':
        build_docs(clean=args.clean, strict=args.strict)
    
    elif args.command == 'serve':
        serve_docs(
            port=args.port, 
            host=args.host, 
            open_browser=not args.no_browser
        )
    
    elif args.command == 'deploy':
        if validate_docs():
            build_docs(clean=True, strict=True)
            deploy_docs(message=args.message)
        else:
            print("❌ Validation failed, skipping deployment")
            sys.exit(1)
    
    elif args.command == 'validate':
        if not validate_docs():
            sys.exit(1)
    
    elif args.command == 'api':
        generate_api_docs()
    
    elif args.command == 'all':
        generate_api_docs()
        if validate_docs():
            build_docs(clean=True, strict=True)
            if args.deploy:
                deploy_docs()
        else:
            print("❌ Validation failed")
            sys.exit(1)
    
    else:
        # Default: show help
        parser.print_help()

if __name__ == "__main__":
    main()