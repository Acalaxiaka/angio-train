@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==== angio-train 安装（只需运行一次，需要联网；之后训练完全离线）====
echo 需要：Windows 10/11 64 位、NVIDIA 显卡驱动 ^>= 531、Python 3.10/3.11/3.12（python.org 安装时勾选 Add to PATH）
set PY=
py -3.11 -V >nul 2>&1 && set PY=py -3.11
if "%PY%"=="" py -3.12 -V >nul 2>&1 && set PY=py -3.12
if "%PY%"=="" py -3.10 -V >nul 2>&1 && set PY=py -3.10
if "%PY%"=="" python -V >nul 2>&1 && set PY=python
if "%PY%"=="" (echo [错误] 未找到 Python。请先安装 Python 3.11: https://www.python.org/downloads/ （国内可用 https://registry.npmmirror.com/binary.html?path=python/ ） & pause & exit /b 1)
if not exist venv\Scripts\python.exe (
  echo [1/4] 创建虚拟环境 venv ...
  %PY% -m venv venv || (echo [错误] 创建 venv 失败 & pause & exit /b 1)
)
set VPY=venv\Scripts\python.exe
set TUNA=-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
%VPY% -m pip install -U pip %TUNA% >nul
echo [2/4] 安装 CUDA 版 PyTorch（约 2.5GB，请耐心）...
%VPY% -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1 && goto torch_ok
%VPY% -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 && goto torch_ok
echo   官方源失败，尝试阿里云镜像 cu121 ...
%VPY% -m pip install torch==2.5.1 torchvision==0.20.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu121/ %TUNA% && goto torch_ok
echo   尝试 cu118（旧驱动）...
%VPY% -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118 && goto torch_ok
%VPY% -m pip install torch==2.5.1 torchvision==0.20.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu118/ %TUNA% && goto torch_ok
echo [错误] PyTorch 安装失败，请检查网络后重试 & pause & exit /b 1
:torch_ok
echo [3/4] 安装其它依赖（清华镜像）...
%VPY% -m pip install -r requirements.txt %TUNA% || %VPY% -m pip install -r requirements.txt || (echo [错误] 依赖安装失败 & pause & exit /b 1)
echo [4/4] 准备 ImageNet 预训练权重 ...
if exist weights\efficientnet_b0.pth (echo   已内置 efficientnet_b0.pth) else (%VPY% download_weights.py --arch efficientnet_b0 --mirror-first)
%VPY% -c "import torch;print('PyTorch',torch.__version__,'CUDA 可用:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '（未检测到显卡，请更新 NVIDIA 驱动）')"
echo ==== 安装完成。可双击 自测.bat 检查，再双击 训练.bat ====
pause
