@echo off
setlocal
cd /d "%~dp0.."
if not defined CXX set "CXX=C:\Cyuyan\mingw64\bin\g++.exe"
"%CXX%" -O3 -std=c++17 -shared -static-libgcc -static-libstdc++ navigation\native\fast_grid.cpp -o navigation\native\fast_grid.dll
exit /b %errorlevel%
