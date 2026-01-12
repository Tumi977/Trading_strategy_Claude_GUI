@echo off
chcp 65001 >nul
echo ==========================================
echo   多维自适应趋势动量交易系统 - GUI
echo ==========================================
echo.

REM 检查虚拟环境是否存在
if not exist "venv\Scripts\activate.bat" (
    echo 正在创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo 错误: 无法创建虚拟环境
        pause
        exit /b 1
    )
)

REM 激活虚拟环境
call venv\Scripts\activate.bat

REM 检查是否需要安装依赖
echo 检查依赖...
pip show PyQt6 >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 错误: 依赖安装失败
        pause
        exit /b 1
    )
)

REM 启动GUI
echo 启动图形界面...
python run_gui.py

REM 如果出错，暂停显示错误信息
if errorlevel 1 (
    echo.
    echo 程序异常退出
    pause
)
