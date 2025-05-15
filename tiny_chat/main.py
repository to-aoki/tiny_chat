import os
import sys
import argparse

# https://discuss.streamlit.io/t/message-error-about-torch/90886/9
# RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_
import torch
torch.classes.__path__ = []

os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

def main():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    parser = argparse.ArgumentParser(description="Streamlit Chat Application")
    parser.add_argument("--database", "-d", action="store_true", help="only database")
    parser.add_argument("--server_mode", "-s", action="store_true", help="server mode (don't save settings to file)")
    args = parser.parse_args()

    # Deployの非表示
    hide_elements_style = """
        <style>
        .stAppDeployButton {
                visibility: hidden;
            }
        </style>
    """

    # データベースモードかチャットモードかを判定
    if args.database:
        import streamlit as st
        # データベースモード
        st.set_page_config(page_title="データベース", layout="wide")  # stの他コンポーネントの利用都合が不透明なので真っ先に呼ぶ
        # CSSを適用
        st.markdown(hide_elements_style, unsafe_allow_html=True)
        from tiny_chat.database.database import run_database_app
        run_database_app()
    else:
        import streamlit as st
        # チャットモード（デフォルト）
        st.set_page_config(page_title="チャット", layout="wide")
        # CSSを適用
        st.markdown(hide_elements_style, unsafe_allow_html=True)
        from tiny_chat.chat.app import run_chat_app
        run_chat_app(server_mode=args.server_mode)


if __name__ == "__main__":
    main()