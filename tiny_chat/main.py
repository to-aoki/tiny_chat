import os
import sys
import argparse


def main():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    parser = argparse.ArgumentParser(description="Streamlit Chat Application")
    parser.add_argument("--database", "-d", action="store_true", help="only database")
    parser.add_argument("--server_mode", "-s", action="store_true", help="server mode (don't save settings to file)")
    args = parser.parse_args()

    # データベースモードかチャットモードかを判定
    if args.database:
        # データベースモード
        import streamlit as st
        st.set_page_config(page_title="データベース", layout="wide")  # stの他コンポーネントの利用都合が不透明なので真っ先に呼ぶ
        from tiny_chat.database.database import run_database_app
        run_database_app()
    else:
        # チャットモード（デフォルト）
        import streamlit as st
        st.set_page_config(page_title="チャット", layout="wide")
        from tiny_chat.chat.app import run_chat_app
        run_chat_app(server_mode=args.server_mode)


if __name__ == "__main__":
    main()