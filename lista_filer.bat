@echo off
setlocal
set "target_dir=C:\Users\jakpel\OneDrive - Kungsbacka kommun\GitHub\ComputerVisionCounter_video"
set "output_file=C:\Users\jakpel\OneDrive - Kungsbacka kommun\GitHub\ComputerVisionCounter_video\folder_structure.txt"
tree "%target_dir%" /f /a > "%output_file%"
echo Struktur av mappen har sparats i %output_file%
endlocal