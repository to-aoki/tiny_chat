import os
import sys
import streamlit.web.cli

def get_app_path():
    return os.path.join(os.path.dirname(__file__), 'app.py')

def main():
    sys.argv = [
        "streamlit",
        "run",
        get_app_path(),
        "--server.address=127.0.0.1"
    ]
    streamlit.web.cli.main()

if __name__ == '__main__':
    main()
