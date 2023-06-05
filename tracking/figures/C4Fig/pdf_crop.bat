@echo off

for /r . %%a in (*.vsdx) do (
	echo "%%a"
	viso2pdf.exe "%%a"
)

::viso2pdf.exe end-to-end.vsdx

::pdfcrop end-to-end.pdf end-to-end.pdf

pause