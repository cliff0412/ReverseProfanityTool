@echo off
SETLOCAL

REM Check Python installation
python --version >nul 2>&1
IF %errorlevel% NEQ 0 (
    echo Python is not installed or not on PATH
    EXIT /B 1
)

REM Check pip installation
pip --version >nul 2>&1
IF %errorlevel% NEQ 0 (
    echo Pip is not installed or not on PATH
    EXIT /B 1
)

REM Check if virtual environment already exists
IF NOT EXIST envir (
    REM Create virtual environment
    python -m venv envir
)

REM Activate virtual environment
CALL envir\Scripts\activate

REM Install Python requirements if not already satisfied
pip list > temp.txt
FOR /F %%i IN (requirements.txt) DO (
    FIND "%%i" temp.txt >nul 2>&1
    IF %errorlevel% NEQ 0 (
        pip install %%i
    )
)
DEL temp.txt

REM Check if Go is installed
go version >nul 2>&1
IF %errorlevel% NEQ 0 (
    echo Go is not installed. Installing...
    choco install golang
)

REM Build and run Go scripts
go mod init seedFileGenerator
go get github.com/ethereum/go-ethereum/crypto/secp256k1
go get github.com/farces/mt19937_64
go build seedFileGenerator\main.go
main.exe

go build sortBinary\main.go
main.exe

REM Deactivate Python virtual environment
CALL deactivate

ENDLOCAL
