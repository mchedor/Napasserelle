@echo off
setlocal

REM === Configuration ===
set PYTHON_VERSION=3.11.9
set "APP_DIR=%LOCALAPPDATA%\Napasserelle"
set "ENV_DIR=%APP_DIR%\venv"

echo ======================================
echo Installation de Napari + Napasserelle
echo ======================================

REM -----------------------------------------------------------------
REM Vérifie Python
REM -----------------------------------------------------------------
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python non trouvé.
    echo Téléchargement...

    powershell -Command ^
      "Invoke-WebRequest -Uri https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe -OutFile python_installer.exe"

    echo Installation silencieuse...
    start /wait python_installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1

    del python_installer.exe
)

echo.
echo Création du venv...

python -m venv "%ENV_DIR%"

call "%ENV_DIR%\Scripts\activate.bat"

python -m pip install --upgrade pip

echo.
echo Installation de Napasserelle ...

pip install "napasserelle[ez_install]==0.0.0"

echo.
echo Création du raccourci Bureau...

powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Napari.lnk');$s.TargetPath='%ENV_DIR%\Scripts\python.exe';$s.Arguments='-m napari';$s.WorkingDirectory='%ENV_DIR%';$s.IconLocation='%ENV_DIR%\Lib\site-packages\napari\resources\icon.ico';$s.Save();"

echo.
echo Installation terminée.
pause