import streamlit as st
from typing import Optional
import ipaddress
import re

def _normalize_ip(s: str) -> Optional[str]:
    """'1.2.3.4', '1.2.3.4:1234', '[2001:db8::1]:1234' 等を IP だけに正規化して返す"""
    s = (s or "").strip()
    if not s:
        return None

    # [IPv6]:port
    m = re.match(r"^\[(?P<ip>[^]]+)\](?::\d+)?$", s)
    if m:
        s = m.group("ip")
    else:
        # IPv4:port (IPv6 は ':' が複数あるので対象外)
        if s.count(":") == 1 and s.split(":")[1].isdigit():
            s = s.split(":", 1)[0].strip()

    try:
        return str(ipaddress.ip_address(s))
    except ValueError:
        return None


def _first_ip_from_xff(xff: str) -> Optional[str]:
    """X-Forwarded-For の左から順に妥当なIPを返す"""
    for part in (xff or "").split(","):
        ip = _normalize_ip(part)
        if ip:
            return ip
    return None


def get_remote_ip() -> str:
    """
    クライアントのIPアドレスを取得する（ベストエフォート）
    - X-Forwarded-For / X-Real-IP 等のヘッダを最優先 (プロキシ対応)
    - 次に Streamlit 1.45+ の st.context.ip_address
    - localhost のときは 127.0.0.1 を返す
    """
    # 1) ヘッダから推定（st.context.headers があれば優先）
    ctx = getattr(st, "context", None)
    headers = getattr(ctx, "headers", None) if ctx else None

    # 古い Streamlit 向け websocket headers（内部API）
    if not headers:
        try:
            from streamlit.web.server.websocket_headers import _get_websocket_headers
            headers = _get_websocket_headers()
        except Exception:
            headers = None

    if headers:
        # st.context.headers はキーが大小文字無視で取れる仕様
        xff = headers.get("x-forwarded-for")
        if xff:
            ip = _first_ip_from_xff(xff)
            if ip:
                return ip

        for k in ("x-real-ip", "cf-connecting-ip", "x-client-ip"):
            v = headers.get(k)
            ip = _normalize_ip(v.split(",")[0] if isinstance(v, str) else str(v))
            if ip:
                return ip

    # 2) 公式の ip_address（1.45+）
    # ヘッダで見つからなかった場合のフォールバック
    ip = getattr(ctx, "ip_address", None) if ctx else None
    
    # DEBUG: 実際の値を確認するためのログ出力
    try:
        from tiny_chat.utils.logger import get_logger
        logger = get_logger()
        logger.info(f"DEBUG: headers={dict(headers) if headers else 'None'}, ctx.ip_address={ip}")
    except Exception as e:
        # ロガー取得失敗時のフォールバック
        import logging
        logging.getLogger(__name__).info(f"DEBUG(fallback): headers={dict(headers) if headers else 'None'}, ctx.ip_address={ip}")

    if ip is not None:
        # docs: localhost アクセス時は None が期待値
        if str(ip).strip():
            return str(ip)

    # 3) ローカル開発（localhost）は docs 上 None が正常
    #    → Unknown より 127.0.0.1 の方が扱いやすいならこちら
    return "127.0.0.1"