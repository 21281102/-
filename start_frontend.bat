@echo off
chcp 65001 >nul
echo 正在启动前端服务...
cd frontend
npm run dev
pause


