from streamlit.components.v1 import html
import uuid


def copy_button(text, button_text="コピー", height=50):
    # 一意のIDを生成
    unique_id = str(uuid.uuid4()).replace("-", "")
    
    # 特殊文字をエスケープ
    text = text.replace('\"', '\\\"').replace('\\n', '\\\\n').replace('\\r', '\\\\r')
    copy_botton_script = f"""
    <div>
        <button 
            id="copyButton_{unique_id}"
            onclick="copyToClipboard_{unique_id}()"
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
        <span id="copyStatus_{unique_id}" style="display: none; margin-left: 10px; font-size: 12px; color: #666;"></span>
    </div>
    
    <script>
        function copyToClipboard_{unique_id}() {{
            const text = `{text}`;
            const copyButton = document.getElementById('copyButton_{unique_id}');
            const copyStatus = document.getElementById('copyStatus_{unique_id}');
            
            // 複数のクリップボード実装方法を試す
            const copyText = async () => {{
                try {{
                    // 方法1: Modern API (HTTPS必須)
                    await navigator.clipboard.writeText(text);
                    showSuccess();
                }} catch (err1) {{
                    try {{
                        // 方法2: フォールバック (非推奨だが広くサポート)
                        const textArea = document.createElement('textarea');
                        textArea.value = text;
                        textArea.style.position = 'fixed';
                        textArea.style.left = '-999999px';
                        textArea.style.top = '-999999px';
                        document.body.appendChild(textArea);
                        textArea.focus();
                        textArea.select();
                        
                        const successful = document.execCommand('copy');
                        document.body.removeChild(textArea);
                        
                        if (successful) {{
                            showSuccess();
                        }} else {{
                            showError('コピーに失敗しました');
                        }}
                    }} catch (err2) {{
                        showError('お使いのブラウザではコピーできません');
                        console.error('Clipboard error:', err2);
                    }}
                }}
            }};
            
            function showSuccess() {{
                // ボタンの色を一時的に変更してフィードバックを提供
                const originalColor = copyButton.style.backgroundColor;
                copyButton.style.backgroundColor = '#45a049';
                copyStatus.textContent = 'コピーしました!';
                copyStatus.style.color = '#45a049';
                copyStatus.style.display = 'inline';
                
                setTimeout(() => {{
                    copyButton.style.backgroundColor = originalColor;
                    copyStatus.style.display = 'none';
                }}, 2000);
            }}
            
            function showError(message) {{
                copyStatus.textContent = message;
                copyStatus.style.color = '#f44336';
                copyStatus.style.display = 'inline';
                
                setTimeout(() => {{
                    copyStatus.style.display = 'none';
                }}, 3000);
            }}
            
            copyText();
        }}
    </script>
    """
    html(copy_botton_script, height=height)
