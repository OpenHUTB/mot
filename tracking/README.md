
# 毕业论文
将`figures`目录下的所有viso文件转换成pdf文件，然后编译`hnumain.tex`。
## 流程
```shell
pdflatex.exe -synctex=1 -interaction=nonstopmode "hnumain".tex
```


## 问题

### Latex
Q: Linux平台编译时候需要使用XeLatex进行编译
Critical ctex error: "fontset-unavailable" CTeX fontset `fandol' is unavailable in current mode. For immediate help type H <return>. }

* A: 换成xlatex编译器

Q: Linux平台编译出现：Undefined control sequence. \makecover

* A：暂时注释掉cover.tex中的%\makecover


### TexStudio

A:
Options -> Configure TeXstudio -> Build -> Default Compiler -> XeLaTex


### 平台
Q：Linux平台编译中文报错
Environment CJK* undefined. ^^I\begin{CJK*}

A: 注释掉\begin{CJK*}{UTF8}{song}

### 参考文献
Q: 参考文献都是问号

A：Texstudio恢复默认设置重新编译好了（原因未知）。


### 参考文献生成不全
Q: 编译时只生成到八十多个

A：运行clean.bat，然后在texstudio中清除，最后重新编译。



### 平台差异
* Linux平台字体不会加粗；
* Windows 会重复生成最后的pdf文件，原因不明，运行6次后还是生成了。
