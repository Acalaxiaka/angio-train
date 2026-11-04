@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (echo 请先双击 setup.bat 安装 & pause & exit /b 1)
set /p DATA=请输入 angio-frames 输出文件夹（含 labels.csv，可直接把文件夹拖进来）: 
set DATA=%DATA:"=%
set /p OUT=请输入训练结果输出文件夹（回车默认 runs\exp1）: 
if "%OUT%"=="" set OUT=runs\exp1
set OUT=%OUT:"=%
set EXTRA=
if exist "%OUT%\last.pt" (
  set /p RES=检测到未完成的训练，是否续训？[Y/n]: 
)
if exist "%OUT%\last.pt" if /i not "%RES%"=="n" set EXTRA=--resume
echo 开始训练（EfficientNet-B0, 384, batch 32, AMP, 内存缓存）。可随时关闭窗口，之后续训。
venv\Scripts\python.exe train.py -d "%DATA%" -o "%OUT%" --cache --multitask %EXTRA% %*
if errorlevel 1 (echo [错误] 训练失败，见上方信息 & pause & exit /b 1)
start "" "%OUT%\misclassified.html"
pause
