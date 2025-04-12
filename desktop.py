from streamlit_desktop_app import start_desktop_app

# on ubuntu 24
#  sudo apt-get install build-essential libgl1-mesa-dev
#  pip install pyqt5 pyqtwebengine
#  pip install pywebview[qt]
#  sudo apt install libcairo2-dev libxt-dev libgirepository-2.0-dev
#  pip install pycairo PyGObject
#  pip install streamlit-desktop-app
start_desktop_app("tiny_chat/main.py", title="Tiny Chat App")
