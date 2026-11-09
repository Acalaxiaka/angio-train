@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==== angio-train 安装（只需运行一次，需要联网；之后训练完全离线）====
echo 需要：Windows 10/11 64 位、NVIDIA 显卡驱动 ^>= 531、Python 3.10/3.11/3.12（python.org 安装时勾选 Add to PATH）
echo.
echo 若当前正在慢速下载（官方源），请：关掉本窗口 → 删除本目录下的 venv 文件夹 → 再双击本脚本。
echo PyTorch 优先走国内镜像（阿里云 → 南大 → 上交 → 官方兜底）。详见 手动下载PyTorch.txt
echo.
set PY=
py -3.11 -V >nul 2>&1 && set PY=py -3.11
if "%PY%"=="" py -3.12 -V >nul 2>&1 && set PY=py -3.12
if "%PY%"=="" py -3.10 -V >nul 2>&1 && set PY=py -3.10
if "%PY%"=="" python -V >nul 2>&1 && set PY=python
if "%PY%"=="" (echo [错误] 未找到 Python。请先安装 Python 3.11: https://www.python.org/downloads/ （国内可用 https://registry.npmmirror.com/binary.html?path=python/ ） & pause & exit /b 1)
if not exist venv\Scripts\python.exe (
  echo [1/4] 创建虚拟环境 venv ...
  %PY% -m venv venv || (echo [错误] 创建 venv 失败 & pause & exit /b 1)
) else (
  echo [1/4] 已有 venv，跳过创建
)
set VPY=venv\Scripts\python.exe
set TUNA=-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
set ALIYUN_PIP=-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
%VPY% -m pip install -U pip %TUNA% >nul 2>&1 || %VPY% -m pip install -U pip %ALIYUN_PIP% >nul 2>&1 || %VPY% -m pip install -U pip >nul 2>&1
echo [2/4] 安装 CUDA 版 PyTorch 2.5.1（约 2.5GB；优先国内镜像）...
%VPY% -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1 && goto torch_ok
REM 若已装了无 CUDA 的 torch，先卸掉再装
%VPY% -c "import torch" >nul 2>&1 && (
  echo   检测到已安装的 torch 无 CUDA，先卸载再重装 ...
  %VPY% -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
)

echo   [1/5] 阿里云 cu121 ...
%VPY% -m pip install "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" -f https://mirrors.aliyun.com/pytorch-wheels/cu121/ %TUNA% && goto torch_check
echo   [2/5] 南京大学 cu121 ...
%VPY% -m pip install "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu121 && goto torch_check
echo   [3/5] 上海交大 cu121 ...
%VPY% -m pip install "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" -f https://mirror.sjtu.edu.cn/pytorch-wheels/cu121/ %TUNA% && goto torch_check
echo   [4/5] 官方 download.pytorch.org cu121（可能很慢，可 Ctrl+C 后改手动下载，见 手动下载PyTorch.txt）...
%VPY% -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 && goto torch_check
echo   [5/5] 阿里云 cu118（旧驱动兼容）...
%VPY% -m pip install "torch==2.5.1+cu118" "torchvision==0.20.1+cu118" -f https://mirrors.aliyun.com/pytorch-wheels/cu118/ %TUNA% && goto torch_check
echo [错误] PyTorch 安装失败。请打开 手动下载PyTorch.txt 用浏览器下 wheel 再装，或检查网络后重试。
pause
exit /b 1

:torch_check
%VPY% -c "import torch,sys;print('  已装 PyTorch',torch.__version__,'CUDA:',torch.cuda.is_available());sys.exit(0 if torch.cuda.is_available() else 1)" || (
  echo   [警告] 已装上但 CUDA 不可用。若驱动 ^>= 531，可删 venv 后重跑；或见 手动下载PyTorch.txt
)
:torch_ok
echo [3/4] 安装其它依赖（清华 → 阿里云）...
%VPY% -m pip install -r requirements.txt %TUNA% || %VPY% -m pip install -r requirements.txt %ALIYUN_PIP% || %VPY% -m pip install -r requirements.txt || (echo [错误] 依赖安装失败 & pause & exit /b 1)
echo [4/4] 准备 ImageNet 预训练权重 ...
if exist weights\efficientnet_b0.pth (echo   已内置 efficientnet_b0.pth) else (%VPY% download_weights.py --arch efficientnet_b0 --mirror-first)
%VPY% -c "import torch;print('PyTorch',torch.__version__,'CUDA 可用:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '（未检测到显卡，请更新 NVIDIA 驱动）')"
echo ==== 安装完成。可双击 自测.bat 检查，再双击 训练.bat ====
pause
