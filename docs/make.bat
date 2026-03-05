@echo off
set SPHINXOPTS=
set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build

if "%1"=="" goto help
if "%1"=="help" goto help
goto run

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:run
%SPHINXBUILD% -b %1 %SOURCEDIR% %BUILDDIR%/%1 %SPHINXOPTS%
goto end

:end
