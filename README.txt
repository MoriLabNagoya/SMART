使用方法
========

把以下两个文件放到你原项目目录：
1. SMART_integrated.py
2. annotation_root.py

并保留原项目已有的：
- gui_imageframe.py
- smartlanguage.py
- resource/ 文件夹及模型文件

运行：
    python SMART_integrated.py

主界面点击 M 按钮，会用独立进程启动 annotation_root.py。
关闭标注窗口不会关闭 SMART 主界面。

本版本还做了：
- 默认结果目录改成程序目录下的 results
- resource/smart.ico 不存在时不再因为图标直接崩溃
- 移除旧 measurement Toplevel 中多余的第二个 mainloop
- 修复 annotation_root 自动保存 *_mask.png 后重新打开找不到 mask 的问题
