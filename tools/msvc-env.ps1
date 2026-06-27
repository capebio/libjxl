# Dot-source to get MSVC x64 + LLVM clang-cl into the current PowerShell session.
$vs = "C:\Program Files\Microsoft Visual Studio\2022\Community"
Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vs -DevCmdArguments "-arch=x64" -SkipAutomaticLocation | Out-Null
$env:PATH = "C:\Program Files\LLVM\bin;" + $env:PATH
