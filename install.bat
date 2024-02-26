@echo off


REM Create virtual environment
python -m venv venv_win

REM Add project directory to the virtual environment's site-packages
echo ..\..\..\src\ > venv_win\Lib\site-packages\tfg.pth

REM Activate the virtual environment
call venv_win\Scripts\activate

REM Install project dependencies
pip install -r .\requirements\dev.txt

REM Deactivate the virtual environment
deactivate