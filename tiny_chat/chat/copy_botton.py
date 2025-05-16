from streamlit.components.v1 import html
import uuid


def copy_button(text, button_text="コピー", height=50):
    # 一意のIDを生成
    unique_id = str(uuid.uuid4()).replace("-", "")

    # 特殊文字をエスケープ (JavaScriptのテンプレートリテラル内で安全に使用するため)
    # バックスラッシュ自体を最初にエスケープすることが重要
    escaped_text = text.replace('\\', '\\\\')
    escaped_text = escaped_text.replace('"', '\\"')
    escaped_text = escaped_text.replace('\n', '\\n')
    escaped_text = escaped_text.replace('\r', '\\r')
    escaped_text = escaped_text.replace('`', '\\`')  # バッククォートのエスケープ
    # テンプレートリテラル内で ${...} 形式の展開を防ぐため、ドル記号もエスケープ
    # text内にJavaScriptの式展開 (${expression}) が含まれない前提
    escaped_text = escaped_text.replace('$', '\\$')

    # Streamlit標準の緑色: #00AB41（メインの緑）または #00CD52（明るい緑）
    copy_button_script = f"""
    <div style="display: flex; align-items: center; margin-top: -4px;">
        <button 
            id="copyButton_{unique_id}"
            onclick="copyToClipboard_{unique_id}()"
            style="background-color: #00AB41; 
                   border: none; 
                   color: white; 
                   padding: 8px 16px; 
                   text-align: center; 
                   text-decoration: none; 
                   display: inline-block; 
                   font-size: 14px; 
                   cursor: pointer; 
                   border-radius: 4px;
                   height: 34px;
                   line-height: 1;"
        >{button_text}</button>
        <span id="copyStatus_{unique_id}" style="display: none; margin-left: 10px; font-size: 12px; color: #666;"></span>
    </div>

    <script>
        function copyToClipboard_{unique_id}() {{
            const textToCopy = `{escaped_text}`; // エスケープ済みのテキストを使用
            const copyButton = document.getElementById('copyButton_{unique_id}');
            const copyStatus = document.getElementById('copyStatus_{unique_id}');

            console.log('copyToClipboard_{unique_id} called for ID: {unique_id}. Text to copy (first 50 chars):', textToCopy.substring(0, 50));

            const copyText = async () => {{
                try {{
                    console.log('Attempting navigator.clipboard.writeText for ID {unique_id}');
                    // navigator.clipboard APIはユーザーのジェスチャー(クリックなど)から直接呼び出される必要がある
                    // また、ページがフォーカスされている必要がある
                    // HTTPS環境でのみ動作するのが一般的
                    if (!navigator.clipboard || !navigator.clipboard.writeText) {{
                        console.warn('navigator.clipboard.writeText API is not available for ID {unique_id}. Falling back.');
                        throw new Error('Clipboard API not available'); // catchブロックに移行させる
                    }}
                    await navigator.clipboard.writeText(textToCopy);
                    showSuccess();
                    console.log('navigator.clipboard.writeText succeeded for ID {unique_id}');
                }} catch (err1) {{
                    console.error('navigator.clipboard.writeText failed for ID {unique_id}:', err1);
                    try {{
                        console.log('Attempting document.execCommand("copy") for ID {unique_id} as fallback.');
                        const textArea = document.createElement('textarea');
                        textArea.value = textToCopy;

                        // スタイルで画面外に隠す
                        textArea.style.position = 'fixed';
                        textArea.style.left = '-999999px';
                        textArea.style.top = '-999999px';
                        textArea.style.opacity = '0'; // 念のため
                        textArea.setAttribute('readonly', ''); // ユーザーによる編集を防ぐ

                        document.body.appendChild(textArea);
                        textArea.focus(); // フォーカスを当てる
                        textArea.select(); // テキストを選択

                        let successful = false;
                        try {{
                            // document.execCommandは非推奨だが、フォールバックとして使用
                            successful = document.execCommand('copy');
                        }} catch (execErr) {{
                            console.error('document.execCommand threw an error for ID {unique_id}:', execErr);
                            successful = false;
                        }}

                        document.body.removeChild(textArea);

                        if (successful) {{
                            showSuccess();
                            console.log('document.execCommand("copy") succeeded for ID {unique_id}');
                        }} else {{
                            showError('コピーに失敗しました (execCommand)');
                            console.error('document.execCommand("copy") returned false or failed for ID {unique_id}');
                        }}
                    }} catch (err2) {{
                        showError('お使いのブラウザではコピーできません (fallback error)');
                        console.error('Fallback copy method (textarea with execCommand) failed for ID {unique_id}:', err2);
                    }}
                }}
            }};

            function showSuccess() {{
                console.log('showSuccess called for ID {unique_id}');
                const originalColor = copyButton.style.backgroundColor;
                copyButton.style.backgroundColor = '#00CD52'; // Streamlitの明るい緑色
                copyStatus.textContent = 'コピーしました!';
                copyStatus.style.color = '#00AB41'; // Streamlitの標準緑色
                copyStatus.style.display = 'inline';

                setTimeout(() => {{
                    if (copyButton) {{ // ボタンがまだ存在する場合のみ
                        copyButton.style.backgroundColor = originalColor;
                    }}
                    if (copyStatus) {{ // ステータス表示がまだ存在する場合のみ
                        copyStatus.style.display = 'none';
                    }}
                }}, 2000);
            }}

            function showError(message) {{
                console.log('showError called for ID {unique_id} with message:', message);
                copyStatus.textContent = message;
                copyStatus.style.color = '#f44336'; // 赤
                copyStatus.style.display = 'inline';

                setTimeout(() => {{
                     if (copyStatus) {{ // ステータス表示がまだ存在する場合のみ
                        copyStatus.style.display = 'none';
                    }}
                }}, 3000);
            }}

            copyText();
        }}
    </script>
    """
    html(copy_button_script, height=height)
