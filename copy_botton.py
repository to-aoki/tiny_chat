from streamlit.components.v1 import html


def copy_button(text, button_text="コピー", height=50):
    # 特殊文字をエスケープ
    text = text.replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
    copy_botton_script = f"""
    <button 
        onclick="navigator.clipboard.writeText(`{text}`)"
        style="background-color: #4CAF50; 
               border: none; 
               color: white; 
               padding: 8px 16px; 
               text-align: center; 
               text-decoration: none; 
               display: inline-block; 
               font-size: 14px; 
               margin: 4px 2px; 
               cursor: pointer; 
               border-radius: 4px;"
    >{button_text}</button>
    """
    html(copy_botton_script, height=height)

