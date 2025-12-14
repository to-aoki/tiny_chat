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
    - Streamlit 1.45+ の st.context.ip_address を最優先
    - 次に X-Forwarded-For / X-Real-IP 等のヘッダ
    - localhost のときは 127.0.0.1 を返す
    """
    # 1) まず公式の ip_address（1.45+）
    ctx = getattr(st, "context", None)
    ip = getattr(ctx, "ip_address", None) if ctx else None
    if ip is not None:
        # docs: localhost アクセス時は None が期待値
        if str(ip).strip():
            return str(ip)
        # ip が空文字等は通常ないが念のため
    else:
        # ip_address が未実装の Streamlit もある
        pass

    # 2) ヘッダから推定（st.context.headers があれば優先）
    headers = getattr(ctx, "headers", None) if ctx else None

    # 3) 古い Streamlit 向け websocket headers（内部API）
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
            ip2 = _first_ip_from_xff(xff)
            if ip2:
                return ip2

        for k in ("x-real-ip", "cf-connecting-ip", "x-client-ip"):
            v = headers.get(k)
            ip2 = _normalize_ip(v.split(",")[0] if isinstance(v, str) else str(v))
            if ip2:
                return ip2

    # 4) ローカル開発（localhost）は docs 上 None が正常
    #    → Unknown より 127.0.0.1 の方が扱いやすいならこちら
    return "127.0.0.1" if ip is None else "Unknown"