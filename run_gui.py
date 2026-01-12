"""
GUI启动入口
多维自适应趋势动量交易系统 - 图形界面

使用方法:
    在虚拟环境中运行:
    python run_gui.py
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """检查依赖是否已安装"""
    missing = []

    try:
        import PyQt6
    except ImportError:
        missing.append("PyQt6")

    try:
        import pyqtgraph
    except ImportError:
        missing.append("pyqtgraph")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    try:
        import akshare
    except ImportError:
        missing.append("akshare")

    if missing:
        print("=" * 50)
        print("缺少以下依赖包:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\n请在虚拟环境中运行以下命令安装:")
        print("  pip install -r requirements.txt")
        print("=" * 50)
        return False

    return True


def main():
    """主函数"""
    if not check_dependencies():
        sys.exit(1)

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont
    from gui.main_window import MainWindow

    # 创建应用
    app = QApplication(sys.argv)

    # 设置应用信息
    app.setApplicationName("多维自适应趋势动量交易系统")
    app.setOrganizationName("Trading Strategy")
    app.setApplicationVersion("1.0.0")

    # 设置默认字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)

    # 设置样式表
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            padding: 5px 15px;
            border: 1px solid #ccc;
            border-radius: 3px;
            background-color: #fff;
        }
        QPushButton:hover {
            background-color: #e0e0e0;
        }
        QPushButton:pressed {
            background-color: #d0d0d0;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 3px;
        }
        QTextEdit {
            border: 1px solid #ccc;
            border-radius: 3px;
        }
        QTableWidget {
            border: 1px solid #ccc;
            gridline-color: #ddd;
        }
        QHeaderView::section {
            background-color: #f0f0f0;
            padding: 5px;
            border: 1px solid #ccc;
            font-weight: bold;
        }
        QTabWidget::pane {
            border: 1px solid #ccc;
            border-radius: 3px;
        }
        QTabBar::tab {
            padding: 8px 15px;
            border: 1px solid #ccc;
            border-bottom: none;
            border-radius: 3px 3px 0 0;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #fff;
            border-bottom: 1px solid #fff;
        }
        QTabBar::tab:!selected {
            background-color: #e0e0e0;
        }
        QProgressBar {
            border: 1px solid #ccc;
            border-radius: 3px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 2px;
        }
    """)

    # 创建并显示主窗口
    window = MainWindow()
    window.show()

    # 运行事件循环
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
