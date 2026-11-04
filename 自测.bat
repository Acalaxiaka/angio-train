@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在用合成数据自测（CPU，约 1 分钟），最后应显示 RESULT: ALL PASS
venv\Scripts\python.exe selftest.py
pause
