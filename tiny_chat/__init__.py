import os
import sys
import argparse
import streamlit.web.cli

__version__ = "0.1.1"


def get_app_path():
    return os.path.join(os.path.dirname(__file__), "main.py")


def run_app(database=False, server=False, host="127.0.0.1", port="8501"):
    app_path = get_app_path()

    if database:
        sys.argv = [
            "streamlit",
            "run",
            app_path,
            f"--server.address={host}",
            f"--server.port={port}",
            "--",
            "--database"
        ]
    elif server:
        sys.argv = [
            "streamlit",
            "run",
            app_path,
            f"--server.address={host}",
            f"--server.port={port}",
            "--",
            "--server_mode"
        ]
    else:
        sys.argv = [
            "streamlit",
            "run",
            app_path,
            f"--server.address={host}",
            f"--server.port={port}"
        ]

    try:
        streamlit.web.cli.main()
    except Exception as e:
        print(f"Streamlit起動エラー: {str(e)}")
        import subprocess
        cmd = ["streamlit", "run", app_path, f"--server.address={host}", f"--server.port={port}"]
        subprocess.run(cmd)


def main():
    """コマンドライン引数を処理するエントリポイント"""
    parser = argparse.ArgumentParser(description="Tiny Chat Application")
    parser.add_argument("--database", "-d", action="store_true", help="Run in database mode")
    parser.add_argument("--server_mode", "-s", action="store_true", help="server mode (don't save settings to file)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host address")
    parser.add_argument("--port", default="8501", help="Server port")
    args = parser.parse_args()

    # 引数に基づいてアプリケーションを実行
    run_app(database=args.database, server=args.server_mode, host=args.host, port=args.port)