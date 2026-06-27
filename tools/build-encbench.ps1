# Compile + link the streaming-encode profiling/timing driver.
$ErrorActionPreference = 'Stop'
. C:\Foo\rcw-gaborish\tools\msvc-env.ps1

$src   = 'C:\Foo\rcw-gaborish'
$build = 'C:\Foo\rcw-gaborish\build-bench'

clang-cl /std:c++17 /O2 /MD /EHsc /DNDEBUG /DJXL_STATIC_DEFINE `
  "/I$src" "/I$src\lib\include" "/I$build\lib\include" `
  "/I$src\third_party\highway" "/I$src\third_party\brotli\c\include" `
  "$src\tools\encode_stream_bench.cc" `
  "/Fe$build\encode_stream_bench.exe" `
  "/Fo$build\encode_stream_bench.obj" `
  /link `
  "$build\lib\jxl-internal.lib" `
  "$build\lib\jxl_cms.lib" `
  "$build\third_party\highway\hwy.lib" `
  "$build\third_party\brotli\brotlienc.lib" `
  "$build\third_party\brotli\brotlidec.lib" `
  "$build\third_party\brotli\brotlicommon.lib"
"ENCBENCH_BUILD_EXIT=$LASTEXITCODE"
