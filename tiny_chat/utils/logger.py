import os
import logging
from logging.handlers import RotatingFileHandler
import datetime


class Logger:
    """
    アプリケーションのログを管理するクラス
    """
    def __init__(self, log_dir="logs", log_level=logging.INFO):
        """
        ロガーを初期化する

        Args:
            log_dir (str): ログファイルを保存するディレクトリ
            log_level (int): ログレベル
        """
        self.log_dir = log_dir
        self.log_level = log_level
        self.logger = None
        self.setup_logger()

    def setup_logger(self):
        """ロガーのセットアップを行う"""
        # ログディレクトリがなければ作成
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 現在の日付を取得してログファイル名に使用
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.log_dir, f"chat_app_{today}.log")

        # ロガーの設定
        self.logger = logging.getLogger("chat_app")
        self.logger.setLevel(self.log_level)

        # ハンドラーがすでに設定されている場合は追加しない
        if not self.logger.handlers:
            # ファイルハンドラー（ローテーション付き、最大10MB、バックアップ5個）
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
            )
            
            # フォーマッターの設定
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            # ロガーにハンドラーを追加
            self.logger.addHandler(file_handler)

    def get_logger(self):
        """設定済みのロガーを取得する"""
        return self.logger

    def info(self, message):
        """INFOレベルのログを記録する"""
        if self.logger:
            self.logger.info(message)

    def error(self, message):
        """ERRORレベルのログを記録する"""
        if self.logger:
            self.logger.error(message)

    def warning(self, message):
        """WARNINGレベルのログを記録する"""
        if self.logger:
            self.logger.warning(message)

    def debug(self, message):
        """DEBUGレベルのログを記録する"""
        if self.logger:
            self.logger.debug(message)

    def critical(self, message):
        """CRITICALレベルのログを記録する"""
        if self.logger:
            self.logger.critical(message)

# シングルトンパターンでロガーインスタンスを提供
_logger_instance = None

def get_logger(log_dir="logs", log_level=logging.INFO):
    """
    シングルトンパターンでロガーインスタンスを取得する

    Args:
        log_dir (str): ログディレクトリ
        log_level (int): ログレベル

    Returns:
        Logger: ロガーインスタンス
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = Logger(log_dir, log_level)
    return _logger_instance
