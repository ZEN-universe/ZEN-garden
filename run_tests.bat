@echo off
if "%~1" equ ":main" (
  shift /1
  goto main
)
cmd /d /c "%~f0" :main %*
Rem This will be executed even if main fails
coverage report -m
exit /b

:main
Rem Here is the main stuff
Rem This is important because without it variables are asigned
Rem Directly when the script is run, if this is set they are
Rem assigned for each line also inside the forloop
Rem SETLOCAL EnableDelayedExpansion
setlocal enableextensions enabledelayedexpansion

Rem we copy everything to the temp dir
Rem This gets the directory name of the batch file, i.e. ZEN-garden
for %%a in ("%~dp0\.") do set "parent=%%~nxa"
Rem Copy everything except the .git folder and data folder
robocopy %~dp0 %temp%\%parent% /E /Z /R:5 /W:5 /TBD /NP /V /NFL /NDL /NJH /NJS /nc /ns /np /XD %~dp0.git
Rem go to the dir
cd %temp%\%parent%
Rem create data dir and copy test cases
robocopy .\tests\testcases\ .\data /E /Z /R:5 /W:5 /TBD /NP /V /NFL /NDL /NJH /NJS /nc /ns /np

Rem erase current coverage
coverage erase

Rem Set the search string for replacement
set search=analysis\[\"dataset\"\]

Rem Cycle through all tests
for /f "tokens=*" %%G in ('dir /b /s /a:d "data\test_*"') do (
  echo Running Test: %%G
  echo ====================================

  Rem Get basename and replace in config, base name will be %%~ni (e.g. test_1a)
  for /F %%i in ("%%G") do (
    Rem Cycle through the file, need to set delims to nothing to keep leading spaces
    for /F "tokens=* delims=" %%A in (data\config.py) do (
      Rem Check if it matches the search
      Rem /R accecpt regex, ^ Line beginning  * undefined amount of spaces
      echo.%%A | findstr /R /C:"^ *%search%" 1>nul
      if errorlevel 1 (
        Rem if we did not find the line, we take the line as is
        set new_line=%%A
      ) ELSE (
        Rem if we have a match we split the string by = to get the leading space (if any)
        for /F "delims==" %%B in ("%%A") do (
          set new_line=%%B= "%%~ni"
        )
      )
      Rem Write to temp file !! because of delayed expansion
      echo !new_line! >> data\config.py.temp
    )
    Rem Remove old config
    DEL data\config.py

    Rem Move temp file
    MOVE data\config.py.temp data\config.py

    Rem run the coverage
    coverage run --source="./model,./preprocess,./postprocess" -a compile.py
  )
)
exit /b
