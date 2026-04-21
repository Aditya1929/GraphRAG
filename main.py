import sys
import os
import uvicorn

# Add backend directory to path so api.py can import its siblings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app", "backend"))

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        reload_dirs=[os.path.join(os.path.dirname(__file__), "app", "backend")],
    )
