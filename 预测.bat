@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (echo 请先双击 setup.bat 安装 & pause & exit /b 1)
set /p MODEL=训练结果文件夹（含 best.pt，回车默认 runs\exp1）: 
if "%MODEL%"=="" set MODEL=runs\exp1
set MODEL=%MODEL:"=%
set /p INP=待预测文件夹（PNG 帧或脱敏 DICOM）: 
set INP=%INP:"=%
set /p OUT=预测结果输出文件夹（回车默认 runs\pred）: 
if "%OUT%"=="" set OUT=runs\pred
set OUT=%OUT:"=%
venv\Scripts\python.exe predict.py -m "%MODEL%" -i "%INP%" -o "%OUT%" %*
start "" "%OUT%"
pause
