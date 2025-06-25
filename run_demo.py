#!/usr/bin/env python3
"""
Quick launcher for the demo app
"""

import subprocess
import sys
import os

# def install_requirements():
#     """Install required packages"""
#     requirements = [
#         'streamlit',
#         'opencv-python',
#         'matplotlib',
#         'pillow',
#         'numpy'
#     ]
    
#     print("📦 Installing required packages...")
#     for package in requirements:
#         try:
#             subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
#             print(f"✅ {package} installed")
#         except:
#             print(f"⚠️ {package} might already be installed")

def launch_demo():
    """Launch the Streamlit demo"""
    print("\n🚀 Launching Nigerian Building Detection Demo...")
    print("🌐 Demo will open in your browser automatically")
    print("📝 Perfect for your presentation!")
    print("\n" + "="*50)
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'mockdemo.py',
            '--server.port', '8501',
            '--server.headless', 'false',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Try running manually:")
        print("streamlit run mock_demo_app.py")

if __name__ == "__main__":
    print("🏠 NIGERIAN BUILDING DETECTION - DEMO LAUNCHER")
    print("=" * 50)
    
    # Install requirements first
    # install_requirements()
    
    # Launch demo
    launch_demo()