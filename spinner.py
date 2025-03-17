import streamlit as st


def spinner():
    circle_spinner_css = """
    <style>
    .spinner {
      display: inline-block;
      width: 1em;
      height: 1em;
      border: 0.15em solid rgba(0, 120, 212, 0.2);
      border-top-color: rgba(0, 120, 212, 1);
      border-radius: 50%;
      animation: spin 1s ease-in-out infinite;
      vertical-align: middle;
      margin-right: 0.5em;
    }
    
    @keyframes spin {
      to {
        transform: rotate(360deg);
      }
    }
    
    .spinner-container {
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      background-color: rgba(255, 255, 255, 0.9);
      border-radius: 10px;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      margin: 10px 0;
      font-size: 16px;
    }
    
    .spinner-text {
      font-weight: bold;
      color: #0078D4;
      display: inline;
    }
    </style>
    """

    with st.container():
        st.markdown(circle_spinner_css, unsafe_allow_html=True)

        # 円形のスピナーとメッセージを表示
        st.markdown(f"""
        <div class="spinner-container">
            <div class="spinner"></div>
            <div class="spinner-text">{st.session_state.status_message}</div>
        </div>
        """, unsafe_allow_html=True)
