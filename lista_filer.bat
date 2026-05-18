@echo off
setlocal

REM Creates a local folder tree listing for this repository.
REM No private or absolute paths are stored in this file.
REM The generated folder_structure.txt file is ignored by Git.

set "target_dir=%~dp0"
set "output_file=%~dp0folder_structure.txt"

tree "%target_dir%" /f /a > "%output_file%"

echo Folder structure has been saved to:
echo %output_file%

endlocal
