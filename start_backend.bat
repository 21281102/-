@echo off
chcp 65001 >nul
echo 正在启动后端服务...
cd backend
python main.py
pause


