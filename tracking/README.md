
# 毕业论文

## 流程

pdflatex.exe -synctex=1 -interaction=nonstopmode "hnumain".tex

修改tomm（1月小修、2月接收）；
neurocomputing（1月大修投出去，2月小修、3月初接受）
cviu(一月底投出去)

四篇确定接受后开始找投简历找工作。

撰写毕业论文（二月底完成初稿，tomm接收后给老师看，3月底定稿）

预答辩（4月5号），然后送审；

需要在web of science上查的到，或者老师证明已经接收了。
送审（全部75分以上，一个75分以下延期3个月，再次送审不过延期1年） 40天

答辩（6月1号之前回来盲审意见并答辩）

六月中领证



# 信息
[MOT博士论文](/data2/whd/win10/doc/paper/doctor/doctor.Data/PDF/2507414993)

[fMRI数据处理论文](/data2/whd/win10/doc/paper/doctor/doctor.Data/PDF/0656740001)

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
