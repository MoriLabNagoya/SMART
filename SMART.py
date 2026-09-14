# -*- coding: utf-8 -*-

# Compatibility for Python/OpenSSL builds that do not support
# hashlib(..., usedforsecurity=False). Some PDF/report dependencies use
# this keyword on newer Python versions. On older environments it raises:
# TypeError: 'usedforsecurity' is an invalid keyword argument for openssl_md5()
import hashlib as _smart_hashlib
from functools import wraps as _smart_wraps

def _smart_install_hashlib_compat():
    try:
        _smart_hashlib.md5(b"", usedforsecurity=False)
        return
    except TypeError:
        pass
    except Exception:
        # If the platform rejects MD5 for another reason, do not mask it.
        return

    for _name in ("md5", "sha1", "sha224", "sha256", "sha384", "sha512"):
        _func = getattr(_smart_hashlib, _name, None)
        if _func is None or getattr(_func, "_smart_usedforsecurity_compat", False):
            continue

        def _make_wrapper(func):
            @_smart_wraps(func)
            def _wrapper(data=b"", *args, **kwargs):
                kwargs.pop("usedforsecurity", None)
                return func(data, *args, **kwargs)
            _wrapper._smart_usedforsecurity_compat = True
            return _wrapper

        setattr(_smart_hashlib, _name, _make_wrapper(_func))

_smart_install_hashlib_compat()

import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import  messagebox
from PIL import Image
import glob
import os
import subprocess
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from gui_imageframe import ImageFrame
import csv
from keras_segmentation.models.segnet import resnet50_segnet
from datetime import datetime
from pathlib import Path
from tkinter import font
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont
import math
from smartlanguage import GUI_LGE
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

# 
#from tkinter import PhotoImage

language_order = ["en", "ja", "zh"]


TOOLTIP_TEXT = {
    "add": {
        "en": "Add an image group",
        "ja": "画像グループを追加",
        "zh": "添加图像组",
    },
    "remove": {
        "en": "Remove the last image group",
        "ja": "最後の画像グループを削除",
        "zh": "删除最后一个图像组",
    },
    "start": {
        "en": "Start image processing",
        "ja": "画像処理を開始",
        "zh": "开始图像处理",
    },
    "batch": {
        "en": "Batch process and generate reports",
        "ja": "一括処理してレポートを作成",
        "zh": "批量处理并生成报告",
    },
    "report": {
        "en": "Generate report",
        "ja": "レポートを作成",
        "zh": "生成报告",
    },
    "settings": {
        "en": "Open settings",
        "ja": "設定を開く",
        "zh": "打开设置",
    },
    "clear": {
        "en": "Clear the image-group list",
        "ja": "画像グループ一覧をクリア",
        "zh": "清空图像组列表",
    },
    "annotation": {
        "en": "Open segmentation annotator",
        "ja": "セグメンテーション注釈ツールを開く",
        "zh": "打开分割标注工具",
    },
    "language": {
        "en": "Switch language",
        "ja": "言語を切り替え",
        "zh": "切换语言",
    },
    "help": {
        "en": "Show help",
        "ja": "ヘルプを表示",
        "zh": "显示帮助",
    },
}


class ToolTip:
    """Small Tk tooltip whose text can change with the current UI language."""

    def __init__(self, widget, text_getter, delay=450):
        self.widget = widget
        self.text_getter = text_getter
        self.delay = delay
        self._after_id = None
        self._window = None
        self._label = None

        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def _schedule(self, _event=None):
        self._cancel_schedule()
        self._after_id = self.widget.after(self.delay, self.show)

    def _cancel_schedule(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def show(self):
        self._after_id = None
        if self._window is not None:
            self.refresh()
            return

        text = str(self.text_getter() or "").strip()
        if not text:
            return

        x = self.widget.winfo_pointerx() + 14
        y = self.widget.winfo_pointery() + 18
        self._window = tk.Toplevel(self.widget)
        self._window.wm_overrideredirect(True)
        self._window.wm_geometry(f"+{x}+{y}")

        self._label = tk.Label(
            self._window,
            text=text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            padx=6,
            pady=3,
        )
        self._label.pack()

    def refresh(self):
        if self._label is not None:
            self._label.config(text=str(self.text_getter() or ""))

    def hide(self, _event=None):
        self._cancel_schedule()
        if self._window is not None:
            try:
                self._window.destroy()
            except tk.TclError:
                pass
        self._window = None
        self._label = None

class SmartApp:
    def __init__(self, root):
        self.root = root
        self.lang_index = 0
        self.curlang = language_order[self.lang_index]
        self.root.title(f"{GUI_LGE['roottitle'][self.curlang]}")
        self.root.geometry("700x650")
        self.root.resizable(False, False)
        
        self.rows = []
        self.resolution = 0
        self.unit = "um"
        self.status_var = tk.StringVar()
        self.Pixel_var = tk.StringVar()
        self.checkbox_var = tk.BooleanVar(value=False)  # 初始值为未选中
        self.big_font = font.Font(size=20)  # 设置字体大小
        # 载入图标
        self.load_icons()

        # 顶部按钮栏
        self.create_top_buttons()

        # 表头说明
        self.create_table_header()

        # Treeview区域
        self.create_tree_area()

        self.create_info_input()
        # 输入框和进度条
        self.create_info_input_and_progress()

        # 状态栏
        self.status = tk.StringVar()
        self.status.set(GUI_LGE['ready'][self.curlang])#準備できた 
        self.status_bar = tk.Label(root, textvariable=self.status, bg='lightgreen', anchor='w')
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.COLORS = [ (255,0,0),(0,0,255),(0,255,255),(0,0,0)]
        self.ANOTATE = [ "Focus","Defocus"]
        self.extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
        self.bLoad = False
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, r"resource\last-ResSeg-trans.hdf5")
        yolo_path = os.path.join(base_dir, r"resource\best.pt")
        self.yolo_model = YOLO(yolo_path)  # load a custom 
        self.segmodel = resnet50_segnet(n_classes=4, input_height=256, input_width=256,btrans = True)
        self.tokenizer = None
        self.llmmodel = None
        self.devices = None
        self._qwen_model_path = None
        self.segmodel.load_weights(model_path)# by_name=False
        
    def predict_mask_from_model(self):
        if not hasattr(self, "__imframe") or self.__imframe is None:
            return
        if not hasattr(self, "index"):
            return
	    
        filepath = self.image_paths[self.index]
        image = Image.open(filepath).convert("RGB")
        img_np = np.array(image)
	    
        # 优先使用当前框选 ROI
        if len(self.__imframe.curROICoord) == 2:
            (x1, y1), (x2, y2) = self.__imframe.curROICoord
            x1, x2 = sorted([int(x1), int(x2)])
            y1, y2 = sorted([int(y1), int(y2)])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_np.shape[1], x2)
            y2 = min(img_np.shape[0], y2)
    
            roi = img_np[y1:y2, x1:x2]
            pred = self.segmodel.predict_segmentation(roi)
            roi_mask = ((pred == 1) | (pred == 2)).astype(np.uint8) * 255
    
            full_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = roi_mask
        else:
            pred = self.segmodel.predict_segmentation(img_np)
            full_mask = ((pred == 1) | (pred == 2)).astype(np.uint8) * 255
    
        self.image_masks[self.index] = full_mask
        self.image_overlay_mode[self.index] = "overlay"
        self.__imframe.set_mask(full_mask)
        self.__imframe.set_display_mode("overlay")
    def region_grow_from_scribble(self):
        if not hasattr(self, "__imframe") or self.__imframe is None:
            return
        if not hasattr(self, "index"):
            return
    
        mask = self.__imframe.get_mask()
        if mask is None:
            messagebox.showinfo("Info", "Please scribble foreground first.")
            return
    
        filepath = self.image_paths[self.index]
        image = Image.open(filepath).convert("RGB")
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
        seeds = np.argwhere(mask > 0)
        if len(seeds) == 0:
            return
    
        seed_mask = np.zeros_like(gray, dtype=np.uint8)
        seed_mask[mask > 0] = 255
    
        # 一个比较简单但实用的替代版：用 scribble 初始化 GrabCut
        grab_mask = np.full(gray.shape, cv2.GC_PR_BGD, np.uint8)
        grab_mask[seed_mask > 0] = cv2.GC_FGD
    
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
    
        rect = (1, 1, gray.shape[1] - 2, gray.shape[0] - 2)
        cv2.grabCut(img_np, grab_mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
    
        out_mask = np.where(
            (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    
        self.image_masks[self.index] = out_mask
        self.image_overlay_mode[self.index] = "overlay"
        self.__imframe.set_mask(out_mask)
        self.__imframe.set_display_mode("overlay")
    def load_icons(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, "resource", "smart.ico")
        if os.path.isfile(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except tk.TclError:
                pass
    def switch_display_mode(self, mode):
        if hasattr(self, "__imframe") and self.__imframe is not None:
            self.__imframe.set_display_mode(mode)
            if hasattr(self, "index"):
                self.image_overlay_mode[self.index] = mode
    
    def clear_current_mask(self):
        if hasattr(self, "__imframe") and self.__imframe is not None:
            self.__imframe.clear_mask()
            if hasattr(self, "index"):
                self.image_masks[self.index] = self.__imframe.get_mask()
    def open_annotation_tool(self):
        """Launch the PySide6 annotation program in a separate process."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, "annotation_root_yolo.py"),
            os.path.join(base_dir, "anaotaion-Root(1).py"),
        ]
        annotation_script = next((p for p in candidates if os.path.isfile(p)), None)

        if annotation_script is None:
            messagebox.showerror(
                "Annotation tool",
                "Annotation program was not found.\n\n"
                "Please place annotation_root_yolo.py in the same folder as this program."
            )
            return

        try:
            process = getattr(self, "annotation_process", None)
            if process is not None and process.poll() is None:
                messagebox.showinfo(
                    "Annotation tool",
                    "The annotation program is already running."
                )
                return

            self.annotation_process = subprocess.Popen(
                [sys.executable, annotation_script],
                cwd=base_dir,
            )
            self.update_status("Segmentation Annotator opened.")
        except Exception as exc:
            messagebox.showerror(
                "Annotation tool",
                f"Failed to open annotation program:\n{exc}"
            )

    def _tooltip_text(self, key):
        translations = TOOLTIP_TEXT.get(key, {})
        return translations.get(self.curlang, translations.get("en", key))

    def _add_tooltip(self, widget, key):
        tooltip = ToolTip(widget, lambda key=key: self._tooltip_text(key))
        self.tooltips.append(tooltip)
        return tooltip

    def create_top_buttons(self):
        frame = tk.Frame(self.root, bg="lightblue")
        frame.pack(fill=tk.X)
        self.tooltips = []

        button_specs = [
            ("+", self.add_row, "add", tk.LEFT, None),
            ("-", self.remove_row, "remove", tk.LEFT, None),
            (">", self.StartProcess, "start", tk.LEFT, None),
            ("B", self.BatchProcessAndReport, "batch", tk.LEFT, None),
            ("R", self.Report, "report", tk.LEFT, None),
            ("S", self.open_settings, "settings", tk.LEFT, None),
            ("C", self.Clear, "clear", tk.LEFT, None),
            ("M", self.open_annotation_tool, "annotation", tk.LEFT, None),
            ("L", self.ChangeLanguage, "language", tk.LEFT, None),
            ("?", self.show_help, "help", tk.RIGHT, "lightgray"),
        ]

        for text, command, tooltip_key, side, bg in button_specs:
            kwargs = {
                "text": text,
                "font": self.big_font,
                "width": 2,
                "height": 1,
                "command": command,
            }
            if bg is not None:
                kwargs["bg"] = bg
            button = tk.Button(frame, **kwargs)
            button.pack(side=side, padx=5)
            self._add_tooltip(button, tooltip_key)
    def Clear(self):
        for row_frame, path_var, extra_entry in self.rows:
            row_frame.destroy()
        self.rows.clear()
    def ChangeLanguage(self):
        
        self.lang_index = (self.lang_index + 1) % len(language_order)
        self.curlang = language_order[self.lang_index]
        self.root.title(f"{GUI_LGE['roottitle'][self.curlang]}")
        self.label_index.config(text = GUI_LGE['index'][self.curlang])
        self.label_folder.config(text=GUI_LGE["folder"][self.curlang])
        self.label_select.config(text=GUI_LGE["select"][self.curlang])
        self.label_compound.config(text=GUI_LGE["compound"][self.curlang])
        self.resolution_text.set(GUI_LGE["resolution"][self.curlang] + f": 　　　　{self.resolution} {self.unit}  / pixel")#解像度
        self.checkbox_text.set(GUI_LGE["checkall"][self.curlang])
        self.resultfolderLabel.set(GUI_LGE["resultfolder"][self.curlang])
        self.status_var.set(GUI_LGE['ready'][self.curlang])
        for tooltip in getattr(self, "tooltips", []):
            tooltip.refresh()
    def create_table_header(self):
        head_frame = tk.Frame(self.root, bg="lightgray")
        head_frame.pack(fill=tk.X)
        self.label_index = tk.Label(head_frame, text=GUI_LGE["index"][self.curlang], width=5, anchor="w")
        self.label_index.pack(side=tk.LEFT, padx=5)
        self.label_folder = tk.Label(head_frame, text=GUI_LGE["folder"][self.curlang], width=50, anchor="w")  # 画像フォルダ
        self.label_folder.pack(side=tk.LEFT, padx=5)

        self.label_select = tk.Label(head_frame, text=GUI_LGE["select"][self.curlang], width=6, anchor="center")  # 選択
        self.label_select.pack(side=tk.LEFT, padx=5)

        self.label_compound = tk.Label(head_frame, text=GUI_LGE["compound"][self.curlang], width=15, anchor="w")  # 化合物名
        self.label_compound.pack(side=tk.LEFT, padx=5)
        
    def create_tree_area(self):
        self.tree_frame = tk.Frame(self.root, bg="white",bd =2,relief="solid")
        self.tree_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.tree_frame, bg="white")
        self.scroll_y = tk.Scrollbar(self.tree_frame, orient="vertical", command=self.canvas.yview)
        self.inner_frame = tk.Frame(self.canvas, bg="white")

        self.inner_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll_y.pack(side="right", fill="y")
    def create_info_input(self):
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, pady=5)

        self.resolution_text = tk.StringVar()
        self.resolution_text.set(f"Resolution: 　　　　{self.resolution} {self.unit}  / pixel")#解像度

        self.resolution_label = tk.Label(bottom_frame, textvariable=self.resolution_text)
        self.resolution_label.pack(side=tk.LEFT, padx=5)
        
        self.checkbox_var = tk.BooleanVar(value=False)  # 初始值为未选中

        self.checkbox_text = tk.StringVar()
        self.checkbox_text.set("Process good stomata")
        # 创建 Checkbutton，绑定变量，并设置说明文字
        checkbox = tk.Checkbutton(bottom_frame, textvariable=self.checkbox_text, variable=self.checkbox_var)#気孔を全部処理
        checkbox.pack(fill=tk.X, pady=5)
        
        
        
    def create_info_input_and_progress(self):
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, pady=5)
        self.resultfolderLabel = tk.StringVar()
        self.resultfolderLabel.set("Result folder")
        tk.Label(bottom_frame, textvariable=self.resultfolderLabel).pack(side=tk.LEFT, padx=5)#結果フォルダ:　
        path_var = tk.StringVar()
        self.resultFolder = tk.Entry(bottom_frame,textvariable=path_var,width=40)
        self.resultFolder.pack(side=tk.LEFT, padx=5)
        
        def selectResult():
            path = filedialog.askdirectory()
            if path:
                path_var.set(path)
        path_var.set(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
        tk.Button(bottom_frame, text="+", command=selectResult).pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(bottom_frame, orient="horizontal", length=150, mode="determinate")
        self.progress.pack(side=tk.LEFT, padx=5)
        self.progress["value"] = 0  # 示例默认值
    
    def add_row(self):
        row_frame = tk.Frame(self.inner_frame, bg="white")
        row_frame.pack(fill=tk.X, pady=2)

        index_label = tk.Label(row_frame, text=str(len(self.rows)+1), width=5, bg="white")
        index_label.pack(side=tk.LEFT, padx=5)

        path_var = tk.StringVar()
        entry_path = tk.Entry(row_frame, textvariable=path_var, width=57)
        entry_path.pack(side=tk.LEFT, padx=5)
        assumCompound = tk.StringVar()
        def choose_file():
            path = filedialog.askdirectory()
            if path:
                current_paths = path_var.get()
                if current_paths:
                   new_paths = current_paths + ',' + path
                else:
                   new_paths = path
                   folder_path = Path(new_paths)
                   assumCompound.set(folder_path.name)
                   
                path_var.set(new_paths)
                

        choose_btn = tk.Button(row_frame, text =GUI_LGE['select'][self.curlang], command=choose_file, width=6)#選択
        choose_btn.pack(side=tk.LEFT, padx=5)

        extra_entry = tk.Entry(row_frame, textvariable=assumCompound, width=15)
        extra_entry.pack(side=tk.LEFT, padx=5)

        self.rows.append((row_frame, path_var, extra_entry))
        self.update_status(f"{GUI_LGE['addnewcompound'][self.curlang]}")#新しい化合物を追加しました。
        

    def remove_row(self):
        if self.rows:
            row = self.rows.pop()
            row[0].destroy()
            self.update_status(f"{GUI_LGE['deletecompound'][self.curlang]}")#既存の化合物を削除しました。
        else:
            self.update_status(f"{GUI_LGE['nocompound'][self.curlang]}")#既存の化合物がないです。
    def keep_largest_component(self,apert):
         apert_uint8 = (apert * 255).astype(np.uint8)
        # 连通区域分析（4邻域）
         num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(apert_uint8, connectivity=4)
    
        # 排除背景（label 0），找到面积最大的连通区域
         if num_labels <= 1:
            return np.zeros_like(apert)  # 没有前景区域
    
         largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # stats[1:]是去除背景后的区域信息
    
         # 创建掩膜，仅保留最大区域
         result = (labels == largest_label).astype(np.uint8)
    
         return result
    def ProcessROI(self, curImage, bb, idx, angle, resolution, c, compound, fname, resultPath):
        """Process one detected stomatal ROI and save ROI/mask/overlay outputs."""
        One_Stat = []
        try:
            x1, y1, x2, y2 = np.asarray(bb[idx], dtype=np.int32)
        except Exception as exc:
            print(f"WARNING: invalid bounding box for {fname}, detection {idx + 1}: {exc}")
            return One_Stat

        height, width = curImage.shape[:2]
        x1 = max(0, min(int(x1), width))
        x2 = max(0, min(int(x2), width))
        y1 = max(0, min(int(y1), height))
        y2 = max(0, min(int(y2), height))
        if x2 <= x1 or y2 <= y1:
            print(f"WARNING: empty ROI for {fname}, detection {idx + 1}: {(x1, y1, x2, y2)}")
            return One_Stat

        roi_bgr = curImage[y1:y2, x1:x2].copy()
        if roi_bgr.size == 0:
            return One_Stat

        try:
            out = self.segmodel.predict_segmentation(roi_bgr)
        except Exception as exc:
            print(f"WARNING: segmentation failed for {fname}, detection {idx + 1}: {exc}")
            return One_Stat

        guard = cv2.resize((out == 2).astype(np.uint8),
                           (roi_bgr.shape[1], roi_bgr.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        apert = cv2.resize((out == 1).astype(np.uint8),
                           (roi_bgr.shape[1], roi_bgr.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        mouse = cv2.resize((out == 3).astype(np.uint8),
                           (roi_bgr.shape[1], roi_bgr.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

        guard = self.keep_largest_component(guard)
        apert = self.keep_largest_component(apert)

        gw, gh, ga = self.AnalyseMorphy(guard, angle)
        aw, ah, aa = self.AnalyseMorphy(apert, angle)
        ma = float(np.sum(mouse))

        label_map = guard * 2 + apert + mouse * 3
        color_mask = np.zeros_like(roi_bgr)
        color_mask[label_map == 1] = (0, 0, 255)
        color_mask[label_map == 2] = (255, 0, 0)
        color_mask[label_map == 3] = (0, 255, 255)
        overlay = cv2.addWeighted(roi_bgr, 0.5, color_mask, 0.5, 0)
        curImage[y1:y2, x1:x2] = overlay

        # Save the same output types as smart-validate.
        stem = f"{Path(fname).stem}_id{idx + 1}"
        roi_dir = os.path.join(resultPath, "ROI", compound)
        mask_dir = os.path.join(resultPath, "MASK", compound)
        overlay_dir = os.path.join(resultPath, "OVERLAY", compound)
        os.makedirs(roi_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)
        self.save_image_with_increment(os.path.join(roi_dir, stem + ".png"), roi_bgr)
        self.save_image_with_increment(os.path.join(mask_dir, stem + ".png"), label_map.astype(np.uint8))
        self.save_image_with_increment(os.path.join(overlay_dir, stem + ".png"), overlay)

        od = 0.0
        if ah > 0:
            od = 4 * aa / (math.pi * ah * ah)

        textx, texty = int(x1), int(y1) - 10
        if x1 < 50:
            textx = int(x2)
        if y1 < 50:
            texty = int(y2) + 10
        if ah == 0:
            cv2.putText(curImage, f'ID {idx + 1}: 0 %',
                        (textx + 10, texty + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)
        else:
            cv2.putText(curImage, f'ID {idx + 1}: {100 * od:.1f} %',
                        (textx, texty + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)

        One_Stat = [
            gw * resolution, gh * resolution, ga * resolution * resolution,
            aw * resolution, ah * resolution, aa * resolution * resolution,
            ma * resolution * resolution, od, compound, idx + 1,
            self.ANOTATE[c] if 0 <= int(c) < len(self.ANOTATE) else f"class_{c}",
            fname, idx + 1
        ]
        return One_Stat

    def Process_OneImage(self, fname, resultPath, resolution, compound):
        StatisticsROIs = []
        with open(fname, 'rb') as f:
            image_data = f.read()
        curImage = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if curImage is None:
            print(f"WARNING: failed to read image: {fname}")
            self.progress["value"] += 1
            self.root.update_idletasks()
            return StatisticsROIs

        try:
            results = self.yolo_model(curImage, conf=0.4, iou=0.2)
        except Exception as exc:
            print(f"WARNING: YOLO inference failed for {fname}: {exc}")
            self.progress["value"] += 1
            self.root.update_idletasks()
            return StatisticsROIs

        if not results or results[0].obb is None or len(results[0].obb) == 0:
            print(f"WARNING: no stomata detected in {fname}")
            self.save_image_with_increment(
                os.path.join(resultPath, compound, os.path.basename(fname)), curImage
            )
            self.progress["value"] += 1
            self.root.update_idletasks()
            return StatisticsROIs

        obb = results[0].obb
        try:
            His = list(obb.xywhr.cpu().numpy())
            clss = list(obb.cls.cpu().numpy().astype(np.int32))
            bb = list(obb.xyxy.cpu().numpy())
        except Exception as exc:
            print(f"WARNING: failed to extract detections for {fname}: {exc}")
            self.progress["value"] += 1
            self.root.update_idletasks()
            return StatisticsROIs

        print(f"INFO: detected {len(clss)} stomata in {fname}")
        for idx, c in enumerate(clss):
            _, _, _, _, angle = His[idx]
            if self.checkbox_var.get() and int(c) != 0:
                continue
            Statistics = self.ProcessROI(
                curImage, bb, idx, angle, resolution, int(c),
                compound, os.path.basename(fname), resultPath
            )
            if Statistics:
                StatisticsROIs.append(Statistics)

        os.makedirs(os.path.join(resultPath, compound), exist_ok=True)
        self.save_image_with_increment(
            os.path.join(resultPath, compound, os.path.basename(fname)), curImage
        )
        self.progress["value"] += 1
        self.root.update_idletasks()
        return StatisticsROIs

    def rotate_image(self,image, angle):
            (height, width) = image.shape[:2]
            center = (width / 2, height / 2)
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            rotated = cv2.warpAffine(image.astype(np.uint8), M, (width, height))
            return rotated
    def AnalyseMorphy(self,img, angle):
        out = self.rotate_image(img, -angle*180/3.1415)
        y_nonzero, x_nonzero = np.nonzero(out)
        if np.size(x_nonzero) and np.size(y_nonzero) > 0:
            min_x, max_x = np.min(x_nonzero), np.max(x_nonzero)
            min_y, max_y = np.min(y_nonzero), np.max(y_nonzero)
        else:
            min_x, max_x = 0,0
            min_y, max_y = 0,0
        x,y = max_x-min_x,max_y-min_y
        if x>y:
            w,h = y,x
        else:
            w,h = x,y
        return w,h, np.sum(img)
    def summarize_with_outlier_filtering(self, datas, compoundList):
        sums = []
    
        for i, d in enumerate(datas):
            d = np.array(d, dtype=float)
            compound_name = compoundList[i]
    
            summary = {
                "Compound": compound_name,
                "raw_max": None,
                "raw_min": None,
                "raw_mean": None,
                "filtered_max": None,
                "filtered_min": None,
                "filtered_mean": None,
                "lower_bound": None,
                "upper_bound": None,
            }
    
            if d.size == 0:
                sums.append(summary)
                continue
    
            summary["raw_max"] = float(np.max(d))
            summary["raw_min"] = float(np.min(d))
            summary["raw_mean"] = float(np.mean(d))
    
            Q1 = np.percentile(d, 25)
            Q3 = np.percentile(d, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
    
            summary["lower_bound"] = float(lower_bound)
            summary["upper_bound"] = float(upper_bound)
    
            filtered = d[(d >= lower_bound) & (d <= upper_bound)]
            if filtered.size > 0:
                summary["filtered_max"] = float(np.max(filtered))
                summary["filtered_min"] = float(np.min(filtered))
                summary["filtered_mean"] = float(np.mean(filtered))
    
            sums.append(summary)
    
        return sums
    def DataStatis(self, AllResults):
        awList = []
        poreSizeList = []
        StomaSizeList = []
        compoundList = []
	    
        for each in AllResults:
            each = [x for x in each if x]
	    
            aw = [item[3] for item in each if len(item) > 0]
            pore = [item[5] for item in each if len(item) > 0]
            stoma = [item[2] + item[5] + item[6] for item in each if len(item) > 0]
	    
            awList.append(aw)
            poreSizeList.append(pore)
            StomaSizeList.append(stoma)
	    
            tmp = [item[8] for item in each if len(item) > 0]
            compound_name = tmp[0] if tmp else "Unknown"
            compoundList.append(compound_name)
	    
        awData = self.summarize_with_outlier_filtering(awList, compoundList)
        poreData = self.summarize_with_outlier_filtering(poreSizeList, compoundList)
        stomData = self.summarize_with_outlier_filtering(StomaSizeList, compoundList)
	    
        aw_bounds = {
            item["Compound"]: (item["lower_bound"], item["upper_bound"])
            for item in awData
        }
	    
        outliers = []
        for each in AllResults:
            for i in each:
                compound_name = i[8]
                lowaw, upaw = aw_bounds.get(compound_name, (None, None))
                if lowaw is None or upaw is None:
                    continue
                if i[3] < lowaw or i[3] > upaw:
                    outliers.append(
                        f"Compound: {i[8]} Filename-index: {i[11]}-{i[12]}: pore width: {i[3]:.2f} {self.unit}"
                    )
	    
        self.resultInfor["outliers"] = outliers
        return awData, poreData, stomData
    def ProcessImages(self,collected,resultPath,resolution):        
        now = datetime.now()
        formatted = now.strftime("%Y-%m-%d-%H-%M")
        resultExcel = os.path.join(resultPath,f"Result-{formatted}.csv")

        AllResults = []
        for item in collected:#process each compound
            raw_path_string = item["path"] #one item is one compound
            compound = item["input"]

            # 将路径字符串按逗号分割成多个路径，并去除首尾空格
            path_list = [p.strip() for p in raw_path_string.split(",") if p.strip()]
            imgs = []
        
            for path in path_list:
               # 对每个单独路径进行处理
               for ext in self.extensions:
                   # 使用 glob 查找匹配的文件
                   imgs.extend(glob.glob(os.path.join(path, ext)))
            OneCompound = []
            for fname in imgs:
                ts = os.path.getctime(fname)
                created_time = datetime.fromtimestamp(ts)
                self.resultInfor["ExperimentDate"] = created_time.strftime("%Y-%m-%d")
                Statistics = self.Process_OneImage(fname,resultPath,resolution,compound)
                if Statistics:
                    for stat in Statistics:
                        #One_Stat = [gw*resolution,gh*resolution,ga*resolution*resolution,aw*resolution,ah*resolution,aa*resolution*resolution,ma*resolution*resolution,od,compound,idx+1,self.ANOTATE[c]]        
                        if len(stat)>0:
                            OneCompound.append(stat)
            AllResults.append(OneCompound)
            
        with open(resultExcel,'a',newline='',encoding='utf-8') as f:
            write = csv.writer(f)
            write.writerow([f"{GUI_LGE['processdate'][self.curlang]}",f"{formatted}"])
            write.writerow([f"{GUI_LGE['resolution'][self.curlang]}",f"{resolution} {self.unit}"])
            self.resultInfor["ProcessDate"] =f"{formatted}"
            
            

            #write.writerow(["Resolution:",f"{resolution} {self.unit}"])
            aw,pore, stoma = self.DataStatis(AllResults)
            write.writerow(["Compound", "Item","Max (all)","Min (all)","Mean (all)","Max (IQR)","Min (IQR)","Mean (IQR)"])
            for one in aw:
                write.writerow([f"{one['Compound']}","Pore width",f"{one['raw_max']:.2f}",f"{one['raw_min']:.2f}",f"{one['raw_mean']:.2f}",f"{one['filtered_max']:.2f}",f"{one['filtered_min']:.2f}",f"{one['filtered_mean']:.2f}"]) 
                self.resultInfor["data_stats"]["Porewidth"].append([f"{one['Compound']}",f"{one['raw_max']:.2f}",f"{one['raw_min']:.2f}",f"{one['raw_mean']:.2f}",f"{one['filtered_max']:.2f}",f"{one['filtered_min']:.2f}",f"{one['filtered_mean']:.2f}"])
            for one in pore:
                write.writerow([f"{one['Compound']}","Pore size",f"{one['raw_max']:.2f}",f"{one['raw_min']:.2f}",f"{one['raw_mean']:.2f}",f"{one['filtered_max']:.2f}",f"{one['filtered_min']:.2f}",f"{one['filtered_mean']:.2f}"]) 
                self.resultInfor["data_stats"]["Poresize"].append([f"{one['Compound']}",f"{one['raw_max']:.2f}",f"{one['raw_min']:.2f}",f"{one['raw_mean']:.2f}",f"{one['filtered_max']:.2f}",f"{one['filtered_min']:.2f}",f"{one['filtered_mean']:.2f}"]) 
            for one in stoma:
                write.writerow([f"{one['Compound']}","Stoma size",f"{one['raw_max']:.2f}",f"{one['raw_min']:.2f}",f"{one['raw_mean']:.2f}",f"{one['filtered_max']:.2f}",f"{one['filtered_min']:.2f}",f"{one['filtered_mean']:.2f}"]) 
                self.resultInfor["data_stats"]["Stomasize"].append([f"{one['Compound']}",f"{one['raw_max']:.2f}",f"{one['raw_min']:.2f}",f"{one['raw_mean']:.2f}",f"{one['filtered_max']:.2f}",f"{one['filtered_min']:.2f}",f"{one['filtered_mean']:.2f}"])
            write.writerow(["FileName","Stoma ID","Type",f"Guard Width ({self.unit})",f"Guard Height ({self.unit})",f"Guard Size ({self.unit}*{self.unit})",f"Aperture Width ({self.unit})",f"Aperture Height ({self.unit})",f"Aperture Size ({self.unit}*{self.unit})",f"Mouse ({self.unit}*{self.unit})","Opening Degree","Compound Name"])
            for one in AllResults:
                for stat in one:
                    write.writerow([stat[11],stat[9],stat[10],stat[0],stat[1],stat[2],stat[3],stat[4],stat[5],stat[6],f"{(stat[7]*100):.1f} %",stat[8]])                        
        return AllResults
            #(awList, "Stomata width (um)", "Stomata width", os.path.join(savefolder,"compare-pore-width"),compoundList)
    def GenerateFigure(self, dataList, ylabel_text, title, savePath, compoundList, isPore):
        """Generate a comparison plot and run the appropriate group test.

        Statistical policy:
        - one group: descriptive plot only;
        - two groups: two-sided Welch's independent-samples t-test;
        - three or more groups: one-way ANOVA followed by Tukey HSD.

        Individual stomatal measurements are used as observations, matching
        the current SMART workflow requested for mock-versus-ABA comparison.
        """
        import matplotlib.font_manager as fm
        from scipy.stats import f_oneway, ttest_ind, tukey_hsd

        base_dir = os.path.dirname(os.path.abspath(__file__))
        if self.lang_index == 0:
            myfont = fm.FontProperties(family='Arial')
        elif self.lang_index == 1:
            myfont = fm.FontProperties(
                fname=os.path.join(base_dir, r"resource\msgothic.ttc")
            )
        else:
            myfont = fm.FontProperties(
                fname=os.path.join(base_dir, r"resource\simhei.ttf")
            )

        # Remove invalid values before plotting and testing.
        clean_data = []
        for group in dataList:
            arr = np.asarray(group, dtype=float)
            arr = arr[np.isfinite(arr)]
            clean_data.append(arr)

        boxprops = dict(linewidth=1)
        whiskerprops = dict(linewidth=1)
        capprops = dict(linewidth=1)
        medianprops = dict(linewidth=1)
        flierprops = dict(
            marker='o', markerfacecolor='red', markersize=2,
            markeredgecolor='black'
        )

        positions = list(range(1, len(compoundList) + 1))
        plt.figure(figsize=(10, 10))

        # Matplotlib < 3.9 uses ``labels`` while newer releases prefer
        # ``tick_labels``.  Try the new spelling first and fall back so the
        # program also works in the older DeepStomata environment.
        boxplot_kwargs = dict(
            positions=positions,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops,
            flierprops=flierprops,
        )
        try:
            plt.boxplot(clean_data, tick_labels=compoundList, **boxplot_kwargs)
        except TypeError as exc:
            if "tick_labels" not in str(exc):
                raise
            plt.boxplot(clean_data, labels=compoundList, **boxplot_kwargs)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        for i, group in enumerate(clean_data):
            if len(group) == 0:
                continue
            x = np.random.normal(loc=positions[i], scale=0.05, size=len(group))
            plt.scatter(x, group, alpha=0.7, s=40)
            max_val = float(np.max(group))
            label_offset = 0.01 * max(abs(max_val), 1.0)
            plt.text(
                positions[i], max_val + label_offset,
                f"n = {len(group)}", ha='center', va='bottom',
                fontsize=9, color='black'
            )

        def p_to_star(p_value):
            if p_value < 0.001:
                return '***'
            if p_value < 0.01:
                return '**'
            if p_value < 0.05:
                return '*'
            return 'ns'

        valid_groups = all(len(group) >= 2 for group in clean_data)

        # Exactly two groups: use Welch's t-test, not ANOVA/Tukey HSD.
        if len(compoundList) == 2:
            g1_name, g2_name = compoundList
            g1, g2 = clean_data

            if valid_groups:
                test = ttest_ind(
                    g1, g2, equal_var=False, alternative='two-sided'
                )
                t_value = float(test.statistic)
                p_value = float(test.pvalue)

                n1, n2 = len(g1), len(g2)
                mean1, mean2 = float(np.mean(g1)), float(np.mean(g2))
                var1, var2 = float(np.var(g1, ddof=1)), float(np.var(g2, ddof=1))
                se2 = var1 / n1 + var2 / n2
                df_denominator = (
                    ((var1 / n1) ** 2) / (n1 - 1)
                    + ((var2 / n2) ** 2) / (n2 - 1)
                )
                df = (se2 ** 2) / df_denominator if df_denominator > 0 else float('nan')
                significant = 'yes' if p_value < 0.05 else 'no'

                test_text = (
                    f"Welch's t-test: t({df:.1f})={t_value:.2f}, "
                    f"p={p_value:.3e}"
                )
                plt.gcf().text(0.1, 0.95, test_text, fontsize=18, ha='left')

                y_max = max(float(np.max(g1)), float(np.max(g2)))
                y_min = min(float(np.min(g1)), float(np.min(g2)))
                y_range = max(y_max - y_min, abs(y_max) * 0.1, 1.0)
                y = y_max + 0.08 * y_range
                h = 0.025 * y_range
                plt.plot([1, 1, 2, 2], [y, y + h, y + h, y], lw=1.5, c='black')
                plt.text(
                    1.5, y + h,
                    f"{p_to_star(p_value)}  p={p_value:.2e}",
                    ha='center', va='bottom', fontsize=16, color='black'
                )

                block = (
                    f"<b>{escape(str(title))}</b><br/>"
                    f"Two-sided Welch's t-test: {escape(str(g1_name))} "
                    f"(n={n1}, mean={mean1:.4g}) vs "
                    f"{escape(str(g2_name))} (n={n2}, mean={mean2:.4g})<br/>"
                    f"- Mean difference ({escape(str(g2_name))} - {escape(str(g1_name))}): "
                    f"{(mean2 - mean1):.4g}<br/>"
                    f"- t statistic: {t_value:.4g}<br/>"
                    f"- Welch degrees of freedom: {df:.2f}<br/>"
                    f"- p-value: {p_value:.4g}<br/>"
                    f"- Significant at alpha = 0.05: {significant}"
                )
            else:
                block = (
                    f"<b>{escape(str(title))}</b><br/>"
                    "Welch's t-test was not performed because one or both "
                    "groups contained fewer than two valid observations."
                )

            previous = self.resultInfor.get('significance_test', '').strip()
            self.resultInfor['significance_test'] = (
                f"{previous}<br/><br/>{block}" if previous else block
            )

        # Three or more groups: retain omnibus ANOVA and Tukey HSD.
        elif len(compoundList) > 2:
            if valid_groups:
                anova_result = f_oneway(*clean_data)
                tukey_result = tukey_hsd(*clean_data)
                anova_text = (
                    f"ANOVA: F={anova_result.statistic:.2f}, "
                    f"p={anova_result.pvalue:.3e}"
                )
                plt.gcf().text(0.1, 0.95, anova_text, fontsize=18, ha='left')

                lines = [
                    f"<b>{escape(str(title))}</b><br/>",
                    "One-way ANOVA followed by Tukey HSD:<br/>",
                    f"- F statistic: {float(anova_result.statistic):.4g}<br/>",
                    f"- ANOVA p-value: {float(anova_result.pvalue):.4g}"
                ]
                pvalues = tukey_result.pvalue
                for i in range(len(compoundList)):
                    for j in range(i + 1, len(compoundList)):
                        lines.append(
                            f"<br/>- {escape(str(compoundList[i]))} vs "
                            f"{escape(str(compoundList[j]))}: "
                            f"adjusted p={float(pvalues[i, j]):.4g}"
                        )
                block = ''.join(lines)
            else:
                block = (
                    f"<b>{escape(str(title))}</b><br/>"
                    "ANOVA/Tukey HSD was not performed because at least one "
                    "group contained fewer than two valid observations."
                )

            previous = self.resultInfor.get('significance_test', '').strip()
            self.resultInfor['significance_test'] = (
                f"{previous}<br/><br/>{block}" if previous else block
            )

        plt.ylim(bottom=0)
        plt.ylabel(ylabel_text, fontproperties=myfont, fontsize=20)
        plt.title(title, fontproperties=myfont, fontsize=20)
        plt.tight_layout(rect=(0, 0, 1, 0.93))
        plt.savefig(savePath)
        self.resultInfor['images'].append(savePath)
        plt.close()

    def _is_mock_control(self, name):
        """Return True when a group name represents a mock/vehicle control."""
        normalized = str(name).strip().lower().replace("_", " ").replace("-", " ")
        control_tokens = (
            "mock", "vehicle", "solvent control", "negative control",
            "untreated", "control", "ctrl"
        )
        return any(token == normalized or token in normalized for token in control_tokens)

    @staticmethod
    def _safe_mean(values):
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _safe_filename_component(name):
        """Convert a group name into a filesystem-safe filename component."""
        import re
        text = str(name).strip()
        text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
        text = re.sub(r"-+", "-", text).strip("-._")
        return text or "unknown"

    def GenerateCompare(self, AllResults, savefolder):
        awList = []
        odList = []
        poreSizeList = []
        StomaSizeList = []
        compoundList = []
        # GenerateFigure appends one statistical result block per measured trait.
        self.resultInfor["significance_test"] = ""

        for each in AllResults:
            each = [x for x in each if x]

            awList.append([item[3] for item in each if len(item) > 0])
            odList.append([item[7] for item in each if len(item) > 0])
            poreSizeList.append([item[5] for item in each if len(item) > 0])
            StomaSizeList.append([
                item[2] + item[5] + item[6] for item in each if len(item) > 0
            ])

            tmp = [item[8] for item in each if len(item) > 0]
            compoundList.append(tmp[0] if tmp else "Unknown")

        # Figures can also be generated for a single group.
        self.GenerateFigure(
            awList,
            f"{GUI_LGE['porewidth'][self.curlang]} ({self.unit})",
            f"{GUI_LGE['porewidth'][self.curlang]}",
            os.path.join(savefolder, "compare-pore-width.png"),
            compoundList,
            True
        )
        self.GenerateFigure(
            poreSizeList,
            f"{GUI_LGE['porearea'][self.curlang]} ({self.unit}*{self.unit})",
            f"{GUI_LGE['porearea'][self.curlang]}",
            os.path.join(savefolder, "compare-pore-area.png"),
            compoundList,
            False
        )
        self.GenerateFigure(
            StomaSizeList,
            f"{GUI_LGE['stomaarea'][self.curlang]} ({self.unit}*{self.unit})",
            f"{GUI_LGE['stomaarea'][self.curlang]}",
            os.path.join(savefolder, "compare-stoma-size.png"),
            compoundList,
            False
        )

        group_count = len(compoundList)
        if group_count == 0:
            self.resultInfor["method"] = "No valid groups were processed."
            self.resultInfor["significance_test"] = "No statistical test was performed."
            self.prompt = ""
            return

        means = []
        for i, name in enumerate(compoundList):
            means.append({
                "name": name,
                "width": self._safe_mean(awList[i]),
                "area": self._safe_mean(poreSizeList[i]),
                "ami": self._safe_mean(odList[i]),
                "n": len(awList[i]),
            })

        # Single-group mode: generate a descriptive report without causal or
        # comparative claims and without significance testing.
        if group_count == 1:
            group = means[0]
            self.resultInfor["method"] = (
                f"Leaves were processed as one group: {group['name']}."
            )
            self.resultInfor["significance_test"] = (
                "Not applicable for this descriptive single-group report."
            )
            self.prompt = self.GenerateSingleGroupReport(
                group["name"], group["width"], group["area"],
                group["ami"], group["n"]
            )
            safe_group = self._safe_filename_component(group["name"])
            self.resultInfor["report_filename"] = f"single-{safe_group}-report.pdf"
            return

        # Two-group mode: detect Mock/vehicle control by name, regardless of
        # the order in which rows were entered.
        if group_count == 2:
            if getattr(self, "force_first_group_as_control", False):
                mock_indices = [0]
            else:
                mock_indices = [
                    i for i, name in enumerate(compoundList)
                    if self._is_mock_control(name)
                ]

            if len(mock_indices) == 1:
                control_idx = mock_indices[0]
                treatment_idx = 1 - control_idx
                control_role_note = "identified from the group name"
            else:
                # Preserve backward compatibility when no unique Mock group can
                # be identified, but explicitly tell the language model that the
                # assignment is based on input order.
                control_idx = 0
                treatment_idx = 1
                control_role_note = "assigned from input order because no unique Mock/control name was detected"

            control = means[control_idx]
            treatment = means[treatment_idx]
            self.resultInfor["method"] = (
                f"Individual stomatal measurements were compared between control "
                f"{control['name']} and treatment {treatment['name']} using a "
                f"two-sided Welch's independent-samples t-test; the control was "
                f"{control_role_note}. A threshold of p < 0.05 was used."
            )
            self.prompt = self.GenerateReport(
                control["name"], treatment["name"],
                control["width"], treatment["width"],
                control["area"], treatment["area"],
                control["ami"], treatment["ami"]
            )
            safe_control = self._safe_filename_component(control["name"])
            safe_treatment = self._safe_filename_component(treatment["name"])
            self.resultInfor["report_filename"] = (
                f"comparison-{safe_control}-{safe_treatment}-report.pdf"
            )
            return

        # More than two groups: provide a descriptive multi-group prompt.
        self.resultInfor["method"] = f"Processed {group_count} groups."
        self.prompt = self.GenerateMultiGroupReport(means)
        safe_names = [self._safe_filename_component(g["name"]) for g in means]
        joined_names = "-".join(safe_names)
        self.resultInfor["report_filename"] = f"multigroup-{joined_names}-report.pdf"

    def GenerateSingleGroupReport(self, compound_name, aperture_width, area, ami, n):
        """Create a fact-locked descriptive prompt for one group.

        The single-group findings report only the observed profile. It does not
        discuss controls, treatment effects, group comparisons, or the absence
        of a comparison group.
        """
        self.report_facts = {
            "mode": "single",
            "group_name": str(compound_name),
            "sample_size": int(n),
            "aperture_width": float(aperture_width),
            "aperture_area": float(area),
            "ami": float(ami),
            "unit": str(self.unit),
        }
        self.report_placeholders = {
            "<GROUP_NAME>": str(compound_name),
            "<SAMPLE_SIZE>": str(int(n)),
            "<APERTURE_WIDTH>": f"{aperture_width:.2f} {self.unit}",
            "<APERTURE_AREA>": f"{area:.2f} {self.unit}²",
            "<AMI_VALUE>": f"{ami:.4f}",
        }

        return """
Write one concise scientific findings paragraph from the fact-locked information below.
Synthesize the measurements into a clear descriptive profile rather than mechanically listing them.
Use natural scientific language that differs from a fixed template.

FACT-LOCKED INFORMATION:
- Group: <GROUP_NAME>
- Sample size: <SAMPLE_SIZE> stomata
- Mean aperture width: <APERTURE_WIDTH>
- Mean aperture area: <APERTURE_AREA>
- Mean aperture morphology index (AMI): <AMI_VALUE>

REQUIRED CONTENT:
- Identify the group and sample size.
- Report aperture width, aperture area, and AMI.
- Explain how the three measurements jointly characterize the observed stomatal aperture profile.
- Identify AMI as the aperture morphology index and a relative stomatal-opening measure.
- Keep the paragraph descriptive and limited to the supplied measurements.

STRICT RULES:
- Preserve every placeholder exactly and use each placeholder exactly once.
- Do not rename, split, omit, or duplicate a placeholder.
- Do not introduce another group, a control, a treatment, a comparison, a directional change,
  statistical significance, efficacy, causality, biological mechanism, or external knowledge.
- Do not convert units or reinterpret AMI as stomatal conductance.
- Return exactly one plain-text paragraph with no heading, bullets, Markdown, or table.
"""

    def GenerateMultiGroupReport(self, groups):
        lines = []
        for group in groups:
            lines.append(
                f"- **{group['name']}** (n={group['n']}): width={group['width']:.2f} "
                f"{self.unit}, area={group['area']:.2f} {self.unit}², AMI={group['ami']:.4f}"
            )
        data_text = "\n".join(lines)
        return f"""
You are a scientific expert summarizing stomatal physiology measurements from multiple groups.

{data_text}

Describe the observed group-level patterns concisely. Only call a group a control when its name
explicitly identifies it as Mock, vehicle, untreated, or control. Do not invent mechanisms or
attribute biological activity to a Mock/vehicle control. Do not claim pairwise significance unless
an explicit statistical result is supplied in the prompt.
"""

    @staticmethod
    def _format_signed(value, digits):
        return f"{value:+.{digits}f}"

    def _build_verified_comparison_findings(self, facts):
        """Build a deterministic findings paragraph from verified numeric facts.

        This paragraph is used as a fallback whenever the language-model draft
        changes a value or unit, invents significance, or adds unsupported
        biological interpretation.
        """
        direction = "lower" if facts["width_change"] < 0 else "higher" if facts["width_change"] > 0 else "unchanged"
        width_action = "decreased" if facts["width_change"] < 0 else "increased" if facts["width_change"] > 0 else "remained unchanged"
        area_action = "decreased" if facts["area_change"] < 0 else "increased" if facts["area_change"] > 0 else "remained unchanged"
        ami_action = "decreased" if facts["ami_change"] < 0 else "increased" if facts["ami_change"] > 0 else "remained unchanged"

        if facts["directions_consistent"]:
            consistency = (
                f"The consistent directions of aperture width, aperture area, and AMI indicate "
                f"{direction} relative stomatal opening in the {facts['treatment_name']} group."
            )
        else:
            consistency = (
                "The three measures did not change in the same direction, so they do not support "
                "a single consistent interpretation of relative stomatal opening."
            )

        return (
            f"Compared with the {facts['control_name']} control, the {facts['treatment_name']} group "
            f"showed {direction} values across the evaluated stomatal-opening measures. "
            f"Mean aperture width {width_action} from {facts['control_width']:.2f} to "
            f"{facts['treatment_width']:.2f} {facts['unit']}, corresponding to a change of "
            f"{facts['width_change']:+.2f} {facts['unit']} ({facts['width_percent']:+.1f}%). "
            f"Mean aperture area {area_action} from {facts['control_area']:.2f} to "
            f"{facts['treatment_area']:.2f} {facts['unit']}², corresponding to a change of "
            f"{facts['area_change']:+.2f} {facts['unit']}² ({facts['area_percent']:+.1f}%). "
            f"AMI {ami_action} from {facts['control_ami']:.4f} to {facts['treatment_ami']:.4f}, "
            f"corresponding to a change of {facts['ami_change']:+.4f} "
            f"({facts['ami_percent']:+.1f}%). {consistency}"
        )

    def _build_verified_single_findings(self, facts):
        """Build a deterministic descriptive paragraph for one group."""
        return (
            f"The {facts['group_name']} group comprised {facts['sample_size']} stomata, with a mean "
            f"stomatal aperture width of {facts['aperture_width']:.2f} {facts['unit']}, a mean aperture "
            f"area of {facts['aperture_area']:.2f} {facts['unit']}², and a mean aperture morphology "
            f"index (AMI) of {facts['ami']:.4f}. Together, these measurements provide an integrated "
            "descriptive profile of stomatal aperture morphology and relative opening within the group."
        )

    def _validate_single_qwen_findings(self, text, facts):
        """Validate a descriptive single-group findings paragraph."""
        import re

        normalized = text.lower().replace("μ", "u").replace("µ", "u")
        problems = []

        required_strings = [
            str(facts["sample_size"]),
            f"{facts['aperture_width']:.2f}",
            f"{facts['aperture_area']:.2f}",
            f"{facts['ami']:.4f}",
        ]
        for value in required_strings:
            if value not in text:
                problems.append(f"missing or altered numeric value: {value}")

        if facts["group_name"].lower() not in normalized:
            problems.append("missing or altered group name")

        if re.search(r"\d,\d", text):
            problems.append("decimal comma or altered numeric formatting")
        if re.search(r"\bmm\b|mm²|mm2|millimet", normalized):
            problems.append("incorrect unit conversion")

        unit_norm = facts["unit"].lower().replace("μ", "u").replace("µ", "u")
        if unit_norm not in normalized:
            problems.append("missing width unit")
        if not re.search(re.escape(unit_norm) + r"\s*(?:²|\^?2)", normalized):
            problems.append("missing area unit")

        forbidden_patterns = {
            "introduced comparison or experimental role": (
                r"\bcontrol\b|\breference group\b|\btreatment\b|\bcompared with\b|"
                r"\bcomparison\b|\bversus\b|\bvs\.?\b"
            ),
            "unsupported directional comparison": (
                r"\bincreased\b|\bdecreased\b|\bhigher\b|\blower\b|"
                r"\bmore open\b|\bless open\b|\bopened\b|\bclosed\b"
            ),
            "unsupported significance claim": r"\bsignificant(?:ly)?\b|\bp[- ]?value\b",
            "unsupported mechanism or causality": (
                r"\bmechanism\b|\bcaused\b|\bresulted in\b|\befficacy\b|"
                r"\bbenefit\b|\bharm\b"
            ),
            "incorrect AMI definition": (
                r"stomatal conductance|aperture mean index|aperture mass index"
            ),
        }
        for label, pattern in forbidden_patterns.items():
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                problems.append(label)

        if not re.search(r"descriptive|profile|characteriz", normalized):
            problems.append("missing descriptive characterization")
        if not re.search(r"jointly|together|collectively|combined|integrated|profile", normalized):
            problems.append("missing integrated single-group interpretation")
        if not re.search(r"aperture morphology index", normalized):
            problems.append("missing AMI definition")
        if not re.search(r"relative stomatal[- ]opening", normalized):
            problems.append("missing AMI interpretation as a relative stomatal-opening measure")

        allowed = set(required_strings)
        numeric_text = text.replace(str(facts["group_name"]), "")
        observed = re.findall(r"(?<![A-Za-z])\d+(?:[\.,]\d+)?", numeric_text)
        allowed_canonical = {
            value.rstrip("0").rstrip(".") if "." in value else value
            for value in allowed
        }
        for value in observed:
            if "," in value:
                problems.append(f"unexpected numeric value: {value}")
                continue
            canonical = value.rstrip("0").rstrip(".") if "." in value else value
            if canonical not in allowed_canonical:
                problems.append(f"unexpected numeric value: {value}")

        return len(problems) == 0, problems

    def _validate_qwen_findings(self, text, facts):
        """Check numerical and semantic fidelity of a Qwen-generated paragraph."""
        import re

        normalized = text.lower().replace("μ", "u").replace("µ", "u")
        problems = []

        # Findings and statistical testing are intentionally separated.
        # Reject any significance/p-value wording here, including boilerplate disclaimers.
        if re.search(
            r"\bsignificant(?:ly)?\b|\bstatistical significance\b|"
            r"\bstatistically significant\b|\bp[- ]?value(?:s)?\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            problems.append("statistical-significance wording belongs in the statistical-test section")

        forbidden_patterns = {
            "unsupported mechanism claim": r"\bknown role\b|\bpromot(?:e|es|ed|ing) stomatal closure\b|\bmechanism of action\b",
            "incorrect AMI definition": r"stomatal conductance|aperture mean index|mass index",
            "unsupported causal language": r"\bcaused\b|\bresulted in\b",
        }
        for label, pattern in forbidden_patterns.items():
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                problems.append(label)

        # Reject unit conversion to millimetres or other unsupported units.
        if re.search(r"\bmm\b|mm²|mm2|millimet", normalized):
            problems.append("incorrect unit conversion")

        # Require all supplied values to appear exactly (allowing an optional plus sign).
        required_strings = [
            f"{facts['control_width']:.2f}", f"{facts['treatment_width']:.2f}",
            f"{abs(facts['width_change']):.2f}", f"{abs(facts['width_percent']):.1f}",
            f"{facts['control_area']:.2f}", f"{facts['treatment_area']:.2f}",
            f"{abs(facts['area_change']):.2f}", f"{abs(facts['area_percent']):.1f}",
            f"{facts['control_ami']:.4f}", f"{facts['treatment_ami']:.4f}",
            f"{abs(facts['ami_change']):.4f}", f"{abs(facts['ami_percent']):.1f}",
        ]
        for value in required_strings:
            if value not in text:
                problems.append(f"missing or altered numeric value: {value}")

        # Require the explicit integrated conclusion specified by the fact-locked prompt.
        overall_direction = facts.get("overall_direction")
        if overall_direction == "lower":
            if not (
                re.search(r"lower relative stomatal[- ]opening", normalized)
                and facts["treatment_name"].lower() in normalized
                and facts["control_name"].lower() in normalized
            ):
                problems.append("missing explicit lower-opening comparative conclusion")
        elif overall_direction == "higher":
            if not (
                re.search(r"higher relative stomatal[- ]opening", normalized)
                and facts["treatment_name"].lower() in normalized
                and facts["control_name"].lower() in normalized
            ):
                problems.append("missing explicit higher-opening comparative conclusion")
        elif overall_direction == "mixed":
            if not (
                re.search(r"mixed|different directions|did not (?:follow|change in) the same direction", normalized)
                and re.search(r"does not support|cannot support|no single overall conclusion|single consistent interpretation", normalized)
            ):
                problems.append("missing explicit mixed-pattern conclusion")

        if not re.search(r"aperture morphology index", normalized):
            problems.append("missing AMI definition")
        if not re.search(r"relative stomatal[- ]opening", normalized):
            problems.append("missing relative stomatal-opening interpretation")

        # Detect suspicious numeric values not present in the source facts.
        allowed = set(required_strings)
        allowed.update({"100", "0", "1"})
        numeric_text = text.replace(str(facts["control_name"]), "")
        numeric_text = numeric_text.replace(str(facts["treatment_name"]), "")
        observed = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", numeric_text)
        for value in observed:
            canonical = value.rstrip("0").rstrip(".") if "." in value else value
            allowed_canonical = {
                v.rstrip("0").rstrip(".") if "." in v else v for v in allowed
            }
            if canonical not in allowed_canonical:
                problems.append(f"unexpected numeric value: {value}")

        return len(problems) == 0, problems

    def GenerateReport(self, control_name, treatment_name,
                       control_width, treatment_width,
                       control_area, treatment_area,
                       control_ami, treatment_ami,
                       filename="stomatal_report.txt"):
        """Create a fact-locked two-group prompt using placeholders."""

        def percent_value(value, baseline):
            if baseline == 0:
                return 0.0
            return ((value - baseline) / baseline) * 100.0

        width_change = treatment_width - control_width
        area_change = treatment_area - control_area
        ami_change = treatment_ami - control_ami
        width_percent = percent_value(treatment_width, control_width)
        area_percent = percent_value(treatment_area, control_area)
        ami_percent = percent_value(treatment_ami, control_ami)

        changes = [width_change, area_change, ami_change]
        nonzero_signs = [1 if x > 0 else -1 for x in changes if abs(x) > 1e-12]
        directions_consistent = len(set(nonzero_signs)) <= 1

        self.report_facts = {
            "mode": "comparison",
            "control_name": str(control_name),
            "treatment_name": str(treatment_name),
            "control_width": float(control_width),
            "treatment_width": float(treatment_width),
            "control_area": float(control_area),
            "treatment_area": float(treatment_area),
            "control_ami": float(control_ami),
            "treatment_ami": float(treatment_ami),
            "width_change": float(width_change),
            "area_change": float(area_change),
            "ami_change": float(ami_change),
            "width_percent": float(width_percent),
            "area_percent": float(area_percent),
            "ami_percent": float(ami_percent),
            "directions_consistent": bool(directions_consistent),
            "unit": str(self.unit),
        }

        self.report_placeholders = {
            "<CONTROL_NAME>": str(control_name),
            "<TREATMENT_NAME>": str(treatment_name),
            "<CONTROL_WIDTH>": f"{control_width:.2f} {self.unit}",
            "<TREATMENT_WIDTH>": f"{treatment_width:.2f} {self.unit}",
            "<WIDTH_CHANGE>": f"{width_change:+.2f} {self.unit}",
            "<WIDTH_PERCENT>": f"{width_percent:+.1f}%",
            "<CONTROL_AREA>": f"{control_area:.2f} {self.unit}²",
            "<TREATMENT_AREA>": f"{treatment_area:.2f} {self.unit}²",
            "<AREA_CHANGE>": f"{area_change:+.2f} {self.unit}²",
            "<AREA_PERCENT>": f"{area_percent:+.1f}%",
            "<CONTROL_AMI>": f"{control_ami:.4f}",
            "<TREATMENT_AMI>": f"{treatment_ami:.4f}",
            "<AMI_CHANGE>": f"{ami_change:+.4f}",
            "<AMI_PERCENT>": f"{ami_percent:+.1f}%",
        }

        if directions_consistent:
            if all(x < -1e-12 for x in changes):
                direction_rule = (
                    "All three measures decrease. Explicitly conclude that the treatment group shows "
                    "lower relative stomatal opening than the reference control."
                )
                self.report_facts["overall_direction"] = "lower"
            elif all(x > 1e-12 for x in changes):
                direction_rule = (
                    "All three measures increase. Explicitly conclude that the treatment group shows "
                    "higher relative stomatal opening than the reference control."
                )
                self.report_facts["overall_direction"] = "higher"
            else:
                direction_rule = (
                    "The three measures are directionally consistent but include unchanged values. "
                    "Describe the pattern accurately without overstating a higher or lower overall opening."
                )
                self.report_facts["overall_direction"] = "unchanged_or_partial"
        else:
            direction_rule = (
                "The three measures change in different directions. Explicitly conclude that the mixed pattern "
                "does not support a single overall conclusion about relative stomatal opening."
            )
            self.report_facts["overall_direction"] = "mixed"

        return f"""
Write one concise scientific findings paragraph from the fact-locked information below.
The paragraph must organize the measurements into a coherent comparison and provide one explicit,
data-supported overall interpretation. Use natural scientific language that differs from a fixed template.

FACT-LOCKED INFORMATION:
- Reference control: <CONTROL_NAME>
- Treatment group: <TREATMENT_NAME>
- Mean aperture width: <CONTROL_WIDTH> to <TREATMENT_WIDTH>
- Width change: <WIDTH_CHANGE> (<WIDTH_PERCENT>)
- Mean aperture area: <CONTROL_AREA> to <TREATMENT_AREA>
- Area change: <AREA_CHANGE> (<AREA_PERCENT>)
- Mean aperture morphology index (AMI): <CONTROL_AMI> to <TREATMENT_AMI>
- AMI change: <AMI_CHANGE> (<AMI_PERCENT>)

REQUIRED INTERPRETATION:
- Identify AMI as the aperture morphology index and a relative stomatal-opening measure.
- {direction_rule}

STRICT RULES:
- Preserve every numerical and measurement placeholder exactly and use each one exactly once.
- Use <CONTROL_NAME> and <TREATMENT_NAME> in the explicit overall comparison whenever possible.
  If generic role labels such as "reference control" and "treatment group" are used instead, the
  program will insert the exact group names after generation.
- Do not rename, split, or duplicate a placeholder.
- Do not mention statistical significance, significance testing, or p-values in this paragraph;
  those results are presented separately in the statistical-test section.
- Do not add any number, causal claim, treatment-efficacy claim, biological mechanism,
  concentration assumption, or external knowledge.
- Do not convert units or reinterpret AMI as stomatal conductance.
- Return exactly one plain-text paragraph with no heading, bullets, Markdown, or table.
"""


    def _materialize_placeholders(self, draft):
        """Insert exact facts while allowing generic group-role wording.

        Measurement placeholders are mandatory and must occur exactly once. For
        two-group findings, Qwen may either retain the group-name placeholders or
        use the generic phrases ``reference control`` and ``treatment group``.
        In the latter case, the exact names are inserted programmatically before
        factual and semantic validation.
        """
        import re

        placeholders = getattr(self, "report_placeholders", {}) or {}
        facts = getattr(self, "report_facts", {}) or {}
        problems = []

        optional_name_tokens = set()
        if facts.get("mode") == "comparison":
            optional_name_tokens = {"<CONTROL_NAME>", "<TREATMENT_NAME>"}

        # All measurement placeholders remain mandatory. Name placeholders are
        # optional in comparison mode because exact names can be inserted into
        # generic role phrases after generation.
        for token in placeholders:
            count = draft.count(token)
            if token in optional_name_tokens:
                if count > 1:
                    problems.append(f"placeholder {token} occurred {count} times")
            elif count != 1:
                problems.append(f"placeholder {token} occurred {count} times")

        observed_tokens = set(re.findall(r"<[A-Z][A-Z0-9_]*>", draft))
        unexpected = sorted(observed_tokens.difference(placeholders))
        if unexpected:
            problems.append("unexpected placeholders: " + ", ".join(unexpected))

        if problems:
            return None, problems

        materialized = draft

        # Replace every placeholder that Qwen retained.
        for token, value in placeholders.items():
            if token in materialized:
                materialized = materialized.replace(token, value)

        if facts.get("mode") == "comparison":
            control_name = str(facts.get("control_name", "")).strip()
            treatment_name = str(facts.get("treatment_name", "")).strip()

            # If Qwen used generic role labels instead of name placeholders,
            # inject the exact names without replacing unrelated text.
            if control_name and control_name.lower() not in materialized.lower():
                patterns = [
                    (r"\bthe reference control\b", f"the {control_name} control"),
                    (r"\breference control\b", f"{control_name} control"),
                    (r"\bthe control group\b", f"the {control_name} control group"),
                    (r"\bcontrol group\b", f"{control_name} control group"),
                ]
                for pattern, repl in patterns:
                    materialized, n = re.subn(
                        pattern, repl, materialized, count=1,
                        flags=re.IGNORECASE,
                    )
                    if n:
                        break

            if treatment_name and treatment_name.lower() not in materialized.lower():
                patterns = [
                    (r"\bthe treatment group\b", f"the {treatment_name} group"),
                    (r"\btreatment group\b", f"{treatment_name} group"),
                    (r"\bthe treated group\b", f"the {treatment_name} group"),
                    (r"\btreated group\b", f"{treatment_name} group"),
                ]
                for pattern, repl in patterns:
                    materialized, n = re.subn(
                        pattern, repl, materialized, count=1,
                        flags=re.IGNORECASE,
                    )
                    if n:
                        break

            # Group names must be present by the time semantic validation runs.
            if control_name and control_name.lower() not in materialized.lower():
                problems.append("could not insert the exact control-group name")
            if treatment_name and treatment_name.lower() not in materialized.lower():
                problems.append("could not insert the exact treatment-group name")

        if problems:
            return None, problems
        return materialized, []

    def save_image_with_increment(self, path, image):
    # 分离目录、文件名和扩展名
        directory, filename = os.path.split(path)
        name, ext = os.path.splitext(filename)

        new_path = path
        count = 1

    
        while os.path.exists(new_path):
            new_filename = f"{name}_{count}{ext}"
            new_path = os.path.join(directory, new_filename)
            count += 1

    
        cv2.imwrite(new_path, image)

    
    def generate_report(self, pdf_path, info, notify=True):
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        import datetime
        from reportlab.lib.enums import TA_CENTER

        styles = getSampleStyleSheet()
        style_h2 = styles["Heading2"]
        style_n = styles["Normal"]
        style_h1_center = ParagraphStyle(
            'H1Center', parent=styles['Heading1'], alignment=TA_CENTER, spaceAfter=6)
        style_n_center = ParagraphStyle(
            'NormalCenter', parent=styles['Normal'], alignment=TA_CENTER)
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        elements = []

        today = datetime.date.today().strftime("%Y-%m-%d")
        elements.append(Paragraph("Report on Stoma Experiment", style_h1_center))
        elements.append(Paragraph(f"Created on: {today}", style_n_center))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("1. Date of Stoma Experiment", style_h2))
        elements.append(Paragraph(f"The stoma experiment was carried out on {escape(str(info['ExperimentDate']))}", style_n))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("2. Date of Image Processing", style_h2))
        elements.append(Paragraph(f"The image processing was carried out on {escape(str(info['ProcessDate']))}", style_n))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("3. Processing Method", style_h2))
        elements.append(Paragraph(info["method"], style_n))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("4. Data Statistics", style_h2))
        index = 1
        for section, rows in info["data_stats"].items():
            unit_text = self.unit if section == "Porewidth" else f"{self.unit}x{self.unit}"
            elements.append(Paragraph(f"4.{index} {section} ({unit_text})", style_h2))
            index += 1
            header = ["Group", "Max", "Min", "Mean", "Max (No Outlier)", "Min (No Outlier)", "Mean (No Outlier)"]
            table = Table([header] + rows, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))

        elements.append(Paragraph("5. Outlier stomata", style_h2))
        outs = info.get("outliers", [])
        if isinstance(outs, (list, tuple)) and outs:
            for item in outs:
                elements.append(Paragraph(escape(str(item)), style_n))
        else:
            elements.append(Paragraph("(none)", style_n))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("6. Significance Test", style_h2))
        elements.append(Paragraph(info.get("significance_test", ""), style_n))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("7. Qwen Prompt", style_h2))
        elements.append(Paragraph(escape(str(self.prompt)).replace("\n", "<br/>"), style_n))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("8. Raw Qwen Draft", style_h2))
        elements.append(Paragraph(escape(str(info.get("raw_qwen_findings", ""))), style_n))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(
            "Validation: " + escape(str(info.get("findings_source", ""))), style_n))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("9. Validated Findings", style_h2))
        elements.append(Paragraph(escape(str(info.get("findings", ""))), style_n))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("10. Template-based Findings", style_h2))
        elements.append(Paragraph(escape(str(info.get("template_findings", ""))), style_n))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("11. Related Figures", style_h2))
        for img_path in info.get("images", []):
            elements.append(Image(img_path, width=400, height=300, kind="proportional"))
            elements.append(Spacer(1, 12))

        doc.build(elements)
        if notify:
            messagebox.showinfo("Information", f"The generated report is saved at: {pdf_path}")

    def _ensure_qwen_loaded(self):
        """Load Qwen once and reuse it for every later report in this app process.

        Returns ``True`` when a new model load was performed and ``False`` when
        the already-loaded tokenizer/model were reused.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.environ.get(
            "SMART_QWEN_MODEL",
            os.path.join(base_dir, "resource", "Qwen2.5-7B-Instruct"),
        )

        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                "Qwen model directory was not found:\n"
                f"{model_path}\n\n"
                "Place Qwen2.5-7B-Instruct in resource/Qwen2.5-7B-Instruct, "
                "or set the SMART_QWEN_MODEL environment variable."
            )

        # Normal path: the SmartApp instance survives for the whole Tk session,
        # so these objects stay resident in RAM/VRAM across StartProcess/Report calls.
        if self.tokenizer is not None and self.llmmodel is not None:
            self.bLoad = True
            self.update_status("Qwen2.5 model ready (reused; no reload).")
            return False

        self.update_status("Loading Qwen2.5 language model (first report only)...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=False,
            local_files_only=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            self.devices = torch.device("cuda")
            self.llmmodel = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
        else:
            self.devices = torch.device("cpu")
            self.llmmodel = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            self.llmmodel.to(self.devices)

        self.llmmodel.eval()
        self.bLoad = True
        self._qwen_model_path = model_path
        self.update_status("Qwen2.5 language model loaded and kept in memory.")
        return True

    def Report(self, pdf_path=None, notify=True):
        """Generate a data-grounded findings paragraph with Qwen2.5-Instruct.

        The loaded language model is reused for subsequent reports. ``pdf_path``
        and ``notify`` are used by the batch-report workflow.
        """
        if not getattr(self, "prompt", "").strip():
            messagebox.showinfo(
                "Warning",
                "No report prompt is available. Please process the images first."
            )
            return

        import re
        import torch

        try:
            self._ensure_qwen_loaded()

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a scientific writing assistant specializing in "
                        "plant physiology. Produce concise, data-grounded findings. "
                        "Copy every supplied number and unit exactly. Never recalculate, "
                        "round, convert units, or introduce new numerical values. Do not "
                        "invent statistical significance, biological mechanisms, causal "
                        "claims, sample sizes, or experimental conditions. AMI means "
                        "aperture morphology index and is not stomatal conductance."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        self.prompt.strip()
                        + "\n\nReturn exactly one plain-text scientific paragraph. "
                          "Do not use headings, bullet points, Markdown, tables, "
                          "backticks, or code fences."
                    ),
                },
            ]

            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )

            model_device = self.llmmodel.get_input_embeddings().weight.device
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

            with torch.inference_mode():
                outputs = self.llmmodel.generate(
                    **inputs,
                    max_new_tokens=180,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.02,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0, input_length:]
            description = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            # Remove formatting artifacts before inserting the text into ReportLab.
            description = re.sub(
                r"```(?:[A-Za-z0-9_+\-]+)?[\s\S]*?```",
                "",
                description,
            )
            description = description.replace("`", "")
            description = re.sub(
                r"^(?:ASSISTANT|Assistant)\s*:\s*",
                "",
                description,
            )
            description = re.sub(r"\s+", " ", description).strip()
            self.resultInfor["raw_qwen_findings"] = description

            materialized, placeholder_problems = self._materialize_placeholders(description)
            facts = getattr(self, "report_facts", None)
            if materialized is None:
                description_for_validation = ""
            else:
                description_for_validation = materialized

            if facts is not None and facts.get("mode") == "single":
                valid, validation_problems = self._validate_single_qwen_findings(description_for_validation, facts)
                if placeholder_problems:
                    valid = False
                    validation_problems.extend(placeholder_problems)
                if len(description_for_validation) < 40:
                    valid = False
                    validation_problems.append("abnormally short response")

                if valid:
                    self.resultInfor["findings"] = description_for_validation
                    self.resultInfor["findings_source"] = "Fact-locked Qwen single-group draft passed placeholder and factual validation"
                else:
                    self.resultInfor["findings"] = self._build_verified_single_findings(facts)
                    self.resultInfor["findings_source"] = (
                        "Verified single-group fallback; Qwen draft rejected: "
                        + "; ".join(validation_problems)
                    )
                    print("WARNING: Qwen single-group findings failed validation:")
                    for problem in validation_problems:
                        print(" -", problem)
                    self.update_status("Qwen single-group draft rejected; verified findings used.")
            elif facts is not None and facts.get("mode") == "comparison":
                valid, validation_problems = self._validate_qwen_findings(description_for_validation, facts)
                if placeholder_problems:
                    valid = False
                    validation_problems.extend(placeholder_problems)
                if len(description_for_validation) < 40:
                    valid = False
                    validation_problems.append("abnormally short response")

                if valid:
                    self.resultInfor["findings"] = description_for_validation
                    self.resultInfor["findings_source"] = "Fact-locked Qwen comparison draft passed placeholder and factual validation"
                else:
                    self.resultInfor["findings"] = self._build_verified_comparison_findings(facts)
                    self.resultInfor["findings_source"] = (
                        "Verified comparison fallback; Qwen draft rejected: "
                        + "; ".join(validation_problems)
                    )
                    print("WARNING: Qwen comparison findings failed validation:")
                    for problem in validation_problems:
                        print(" -", problem)
                    self.update_status("Qwen comparison draft rejected; verified findings used.")
            else:
                # Multi-group mode currently receives a less rigid descriptive check.
                if len(description) < 40:
                    raise RuntimeError(
                        "Qwen returned an abnormally short response. Verify that the "
                        "tokenizer and model files come from the same checkpoint."
                    )
                self.resultInfor["findings"] = description
                self.resultInfor["findings_source"] = "Qwen multi-group draft"

            if facts is not None and facts.get("mode") == "single":
                self.resultInfor["template_findings"] = self._build_verified_single_findings(facts)
            elif facts is not None and facts.get("mode") == "comparison":
                self.resultInfor["template_findings"] = self._build_verified_comparison_findings(facts)
            else:
                self.resultInfor["template_findings"] = "Not available for multi-group mode."
            self.resultInfor["validation_errors"] = validation_problems if 'validation_problems' in locals() else []
            self.resultInfor["fallback_used"] = "fallback" in self.resultInfor.get("findings_source", "").lower()

            if pdf_path is None:
                resultFolder = self.resultFolder.get().strip()
                report_filename = self.resultInfor.get(
                    "report_filename",
                    "stomatal-report.pdf",
                )
                pdf_path = os.path.join(resultFolder, report_filename)
            os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)
            self.generate_report(pdf_path, self.resultInfor, notify=notify)
            self.update_status("Report generated.")

        except Exception as e:
            self.update_status("Report generation failed.")
            messagebox.showerror("Report generation error", str(e))


    def _new_result_info(self):
        """Return a clean report-state dictionary for one report case."""
        return {
            "ExperimentDate": "",
            "ProcessDate": "",
            "method": "",
            "data_stats": {
                "Porewidth": [],
                "Poresize": [],
                "Stomasize": [],
            },
            "outliers": [],
            "significance_test": "",
            "findings": "",
            "raw_qwen_findings": "",
            "template_findings": "",
            "findings_source": "",
            "validation_errors": [],
            "fallback_used": False,
            "report_filename": "stomatal-report.pdf",
            "images": [],
        }

    def _populate_case_statistics(self, all_results):
        """Populate PDF summary tables and outlier records for one subset."""
        aw, pore, stoma = self.DataStatis(all_results)
        for one in aw:
            self.resultInfor["data_stats"]["Porewidth"].append([
                str(one["Compound"]), f'{one["raw_max"]:.2f}',
                f'{one["raw_min"]:.2f}', f'{one["raw_mean"]:.2f}',
                f'{one["filtered_max"]:.2f}', f'{one["filtered_min"]:.2f}',
                f'{one["filtered_mean"]:.2f}'
            ])
        for one in pore:
            self.resultInfor["data_stats"]["Poresize"].append([
                str(one["Compound"]), f'{one["raw_max"]:.2f}',
                f'{one["raw_min"]:.2f}', f'{one["raw_mean"]:.2f}',
                f'{one["filtered_max"]:.2f}', f'{one["filtered_min"]:.2f}',
                f'{one["filtered_mean"]:.2f}'
            ])
        for one in stoma:
            self.resultInfor["data_stats"]["Stomasize"].append([
                str(one["Compound"]), f'{one["raw_max"]:.2f}',
                f'{one["raw_min"]:.2f}', f'{one["raw_mean"]:.2f}',
                f'{one["filtered_max"]:.2f}', f'{one["filtered_min"]:.2f}',
                f'{one["filtered_mean"]:.2f}'
            ])

    def _prepare_report_case(self, subset_results, case_folder, experiment_date, process_date, force_first=False):
        """Prepare figures, statistics, prompt, and metadata for one PDF case."""
        self.prompt = ""
        self.report_facts = None
        self.report_placeholders = {}
        self.resultInfor = self._new_result_info()
        self.resultInfor["ExperimentDate"] = experiment_date
        self.resultInfor["ProcessDate"] = process_date
        os.makedirs(case_folder, exist_ok=True)
        self._populate_case_statistics(subset_results)
        self.force_first_group_as_control = bool(force_first)
        try:
            self.GenerateCompare(subset_results, case_folder)
        finally:
            self.force_first_group_as_control = False

    def BatchProcessAndReport(self):
        """Process all groups once and create single and first-vs-rest PDFs.

        Row 1 is always treated as the reference group. For rows 2..N, the
        workflow creates comparison PDFs 1-vs-2, 1-vs-3, ..., and also
        creates one single-group PDF for every row. Qwen is loaded once and
        reused for all generated reports during the current application run.
        """
        if len(self.rows) == 0:
            messagebox.showinfo("Warning", "Please select at least one image folder.")
            return
        if self.resolution == 0:
            messagebox.showinfo("Warning", "Please set the image resolution.")
            return
        result_folder = self.resultFolder.get().strip()
        if not result_folder:
            messagebox.showinfo("Warning", "Please select a result folder.")
            return
        os.makedirs(result_folder, exist_ok=True)

        collected = []
        for row_index, (_, path_var, extra_entry) in enumerate(self.rows, start=1):
            path_value = path_var.get().strip()
            group_name = extra_entry.get().strip() or f"Group{row_index}"
            if not path_value:
                messagebox.showerror("Batch processing", f"Row {row_index} has no image folder.")
                return
            collected.append({"path": path_value, "input": group_name})
            os.makedirs(os.path.join(result_folder, group_name), exist_ok=True)

        total_images = 0
        for item in collected:
            for folder in [p.strip() for p in item["path"].split(",") if p.strip()]:
                for ext in self.extensions:
                    total_images += len(glob.glob(os.path.join(folder, ext)))
        self.progress["maximum"] = max(total_images, 1)
        self.progress["value"] = 0

        self.resultInfor = self._new_result_info()
        self.update_status("Processing all image groups...")
        all_results = self.ProcessImages(collected, result_folder, float(self.resolution))
        experiment_date = self.resultInfor.get("ExperimentDate", "")
        process_date = self.resultInfor.get("ProcessDate", "")

        reports_root = os.path.join(result_folder, "batch_reports")
        singles_root = os.path.join(reports_root, "single")
        comparisons_root = os.path.join(reports_root, "comparisons")
        os.makedirs(singles_root, exist_ok=True)
        os.makedirs(comparisons_root, exist_ok=True)

        generated_paths = []
        try:
            # One single-group report per input row.
            for index, group_results in enumerate(all_results):
                group_name = collected[index]["input"]
                safe_name = self._safe_filename_component(group_name)
                case_folder = os.path.join(singles_root, safe_name)
                self.update_status(f"Generating single report: {group_name}")
                self._prepare_report_case(
                    [group_results], case_folder, experiment_date, process_date,
                    force_first=False
                )
                pdf_path = os.path.join(case_folder, f"single-{safe_name}-report.pdf")
                self.Report(pdf_path=pdf_path, notify=False)
                generated_paths.append(pdf_path)

            # Row 1 versus every later row.
            if len(all_results) >= 2:
                reference_name = collected[0]["input"]
                safe_reference = self._safe_filename_component(reference_name)
                for index in range(1, len(all_results)):
                    treatment_name = collected[index]["input"]
                    safe_treatment = self._safe_filename_component(treatment_name)
                    case_folder = os.path.join(
                        comparisons_root, f"{safe_reference}-vs-{safe_treatment}"
                    )
                    self.update_status(
                        f"Generating comparison: {reference_name} vs {treatment_name}"
                    )
                    self._prepare_report_case(
                        [all_results[0], all_results[index]], case_folder,
                        experiment_date, process_date, force_first=True
                    )
                    pdf_path = os.path.join(
                        case_folder,
                        f"comparison-{safe_reference}-{safe_treatment}-report.pdf"
                    )
                    self.Report(pdf_path=pdf_path, notify=False)
                    generated_paths.append(pdf_path)

            self.update_status(f"Batch reports generated: {len(generated_paths)} PDFs")
            messagebox.showinfo(
                "Batch processing complete",
                f"Generated {len(generated_paths)} PDF reports.\n\nSaved under:\n{reports_root}"
            )
        except Exception as exc:
            self.update_status("Batch report generation failed.")
            messagebox.showerror("Batch report generation error", str(exc))

    def StartProcess(self):
        # Clear report state so a previous comparison cannot leak into a new run.
        self.prompt = ""
        self.report_facts = None
        self.report_placeholders = {}
        self.resultInfor = self._new_result_info()
        resultFolder = self.resultFolder.get()
        if len(self.rows) ==0:
            #messagebox.showinfo("Warning","Please select images")#警告","画像を選択してください！
            messagebox.showinfo(f"{GUI_LGE['warning'][self.curlang]}",f"{GUI_LGE['selectimage'][self.curlang]}")
            return
        if self.resolution == 0:
           #messagebox.showinfo("Warning","Please set the image resolution")#警告","解像度を設定してください！
           messagebox.showinfo(f"{GUI_LGE['warning'][self.curlang]}",f"{GUI_LGE['resolutionset'][self.curlang]}")
           return
        if resultFolder =="":
           #messagebox.showinfo("Warning","Please set the result folder")#警告","結果フォルダを選択してください！
           messagebox.showinfo(f"{GUI_LGE['warning'][self.curlang]}",f"{GUI_LGE['selectfolder'][self.curlang]}")
           return
        if os.path.exists(resultFolder) ==False:
           os.makedirs(resultFolder)
        
        collected = []
        for _, path_var, extra_entry in self.rows:
            collected.append({
                "path": path_var.get(),
                "input": extra_entry.get()
            })
            compound = extra_entry.get()
            if compound:
                compoundPath = os.path.join(resultFolder,compound)
                if os.path.exists(compoundPath) ==False:
                   os.makedirs(compoundPath)
        total_images = 0
        for item in collected:
            path_list = [p.strip() for p in item["path"].split(",") if p.strip()]
            for path in path_list:
                for ext in self.extensions:
                    total_images += len(glob.glob(os.path.join(path, ext)))
        
        self.progress["maximum"] = max(total_images, 1)
        self.progress["value"] = 0
        AllResults = self.ProcessImages(collected,resultFolder,float(self.resolution))
        
        self.update_status(f"{GUI_LGE['finished'][self.curlang]}")
        self.GenerateCompare(AllResults,resultFolder)
        self.update_status(f"{GUI_LGE['saved'][self.curlang]}")
        
        messagebox.showinfo(f"{GUI_LGE['finish'][self.curlang]}",f"{GUI_LGE['finished'][self.curlang]}")#完成","気孔の計測が完了しました。
    def OpenImage(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            #ImageDisplayWindow(path)
            pillimg= Image.open(path)
            top = tk.Toplevel()
            top.geometry("1500x1000")
            top.rowconfigure(0, weight=1)
            top.columnconfigure(0, weight=1)
            #top.columnconfigure(0, weight=0)
            #top.columnconfigure(1, weight=1)
            
            __placeholder = ttk.Frame(top,borderwidth=5, relief="solid")
            __placeholder.grid(row=0, column=0, sticky='nswe')
            __placeholder.rowconfigure(0, weight=1)  # make grid cell expandable
            __placeholder.columnconfigure(0, weight=1)
            #__placeholder.columnconfigure(0, weight=0)  # make grid cell expandable
            #__placeholder.columnconfigure(1, weight=1)

            self.__imframe = ImageFrame(placeholder=__placeholder, roi_size=pillimg.size,curImg = pillimg,status_var = self.status_var, resolu = self.resolution, unit = self.unit, Pixel_var = self.Pixel_var)
            
            self.status_var.set(GUI_LGE['ready'][self.curlang])  # 初始状态消息
            self.status_label = tk.Label(top, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#f0f0f0")
            self.status_label.grid(row=1, column=0, sticky="ew")
    def __ScaleSetted(self):
        self.resolution = float(self.txtPhy.get()) / float(self.txtPixel.get())
        
        self.unit = self.cb.get()     
        self.resolution_text.set(GUI_LGE["resolution"][self.curlang] + f": {self.resolution:.2f} {self.unit} / pixel")#解像度
        self.status.set(GUI_LGE["setresolution"][self.curlang])
        self.scaleWindow.destroy()
    def display_selected_stoma(self, event):
        selection = self.listboxStoma.curselection()
        if selection:
            self.__imframe.curIdx = selection[0]
            self.__imframe.curROICoord = self.__imframe.measuredROI[selection[0]]
            self.__imframe.ReFresh(event)
    def Num2String(self,cnt):
            text =f"The {cnt+1}-th stoma"
            if cnt ==0:
               text = "The first stoma"
            elif cnt ==1:
                text = "The second stoma"
            elif cnt ==2:
                 text = "The third stoma"
            return text            
    def display_selected_image(self, event):
        selection = self.listboxFile.curselection()
        if selection:
            #if len(self.__imframe.curROICoord)==2:
            #    self.__imframe.measuredROI.append(self.__imframe.curROICoord)
            #    self.measuredStoma[self.index] = self.__imframe.measuredROI.copy()
            
            self.index = selection[0]
            filepath = self.image_paths[self.index]

            image = Image.open(filepath).convert("RGB")
            self.__imframe = ImageFrame(
                   placeholder=self.__placeholder,
                   roi_size=image.size,
                   curImg=image,
                   status_var=self.status_var,
                   resolu=self.resolution,
                   unit=self.unit,
                   Pixel_var=self.Pixel_var
                   )
            if self.image_masks[self.index] is not None:
                 self.__imframe.set_mask(self.image_masks[self.index])
            display_mode = self.image_overlay_mode[self.index] if self.image_overlay_mode[self.index] else "raw"
            self.__imframe.set_display_mode(display_mode)
            if len(self.measuredStoma[self.index])>0:
                self.__imframe.SetROI(self.measuredStoma[self.index])
            self.__imframe.ReFresh(event=None)
            
            self.listboxStoma.delete(0, tk.END)
            stoma = len(self.__imframe.measuredROI)
            for i in range(stoma):
               self.listboxStoma.insert(tk.END, self.Num2String(i))#第{i+1}番目気孔
            
    def clear_list(self):
        self.listboxFile.delete(0, tk.END)     
        self.listboxStoma.delete(0, tk.END)    
        self.image_paths.clear()            
    def auto_segment_current_roi(self):
        if not hasattr(self, "__imframe") or self.__imframe is None:
            return
        if len(self.__imframe.curROICoord) != 2:
            messagebox.showinfo("Info", "Please select one ROI first.")
            return
        if not hasattr(self, "index"):
            return
        filepath = self.image_paths[self.index]
        image = Image.open(filepath).convert("RGB")
        image_np = np.array(image)
        (x1, y1), (x2, y2) = self.__imframe.curROICoord
        x1, x2 = sorted([int(x1), int(x2)])
        y1, y2 = sorted([int(y1), int(y2)])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_np.shape[1], x2)
        y2 = min(image_np.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return
        roi = image_np[y1:y2, x1:x2]
        pred = self.segmodel.predict_segmentation(roi)

        # 例如把 aperture/guard 合并成一个前景 mask
        roi_mask = ((pred == 1) | (pred == 2)).astype(np.uint8)
	    
        full_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = roi_mask * 255
	    
        self.image_masks[self.index] = full_mask
        self.image_overlay_mode[self.index] = "overlay"
	    
        self.__imframe.set_mask(full_mask)
        self.__imframe.set_display_mode("overlay")
        self.__imframe.ReFresh()
    def measurement(self):
        if self.resolution ==0:
            messagebox.showinfo(GUI_LGE['warning'][self.curlang],GUI_LGE['resolutionset'][self.curlang])#警告","解像度を設定してください！
            return
        self.image_paths = []
        self.measuredStoma = []
        self.image_masks = []
        self.image_overlay_mode = []
        def open_folder():
            
            folder_path = filedialog.askdirectory()
            self.clear_list()
            if folder_path:
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        full_path = os.path.normpath(os.path.join(folder_path, filename))
                        self.image_paths.append(full_path)
                        self.listboxFile.insert(tk.END, filename)
                self.measuredStoma = [[] for _ in range(len(self.image_paths))]
                self.image_masks = [None for _ in range(len(self.image_paths))]
                self.image_overlay_mode = ["raw" for _ in range(len(self.image_paths))]
        from typing import Any, Optional
        def _to_float(s: Optional[str], default: float = 0.0) -> float:
            try:
                return float(s.strip()) if s is not None else default
            except Exception:
                return default
        def load_measured_xml(xml_path):
            """
            读取由你的保存代码生成的 XML：
            返回 (image_paths, resolution, measuredStoma)
        
            - image_paths: 按出现顺序的图片路径列表
            - resolution: 若所有分辨率相同则为一个 float，否则为与 image_paths 对齐的列表
            - measuredStoma: 与 image_paths 对齐的二维列表，每张图对应一组测量
            """
            tree = ET.parse(xml_path)
            root = tree.getroot()
        
            image_paths = []
            resolutions = []
            measuredStoma= []
        
            current_idx = -1  # 当前正在填充的图片索引
        
            # 按文档顺序扫描节点：imagePath -> Resolution -> Measure(多个) -> imagePath -> ...
            for child in root:
                tag = child.tag
        
                if tag == "imagePath":
                    image_paths.append(child.text or "")
                    measuredStoma.append([])      # 为该图片新建测量列表
                    current_idx += 1
        
                elif tag == "Resolution":
                    # 与 image_paths 顺序对应，允许缺省
                    resolutions.append(_to_float(child.text, default=None))
        
                elif tag == "Measure":
                    # 你的保存代码每个 <Measure> 里建一个 <item>
                    item = child.find("item")
                    if item is None:
                        continue
                    ltx = _to_float(item.findtext("ltx"))
                    lty = _to_float(item.findtext("lty"))
                    rbx = _to_float(item.findtext("rbx"))
                    rby = _to_float(item.findtext("rby"))
        
                    # 保险：若意外在出现第一个 imagePath 前遇到 Measure，则先开一组
                    if current_idx == -1:
                        image_paths.append("")
                        measuredStoma.append([])
                        resolutions.append(None)
                        current_idx = 0
        
                    measuredStoma[current_idx].append([[ltx, lty], [rbx, rby]])
        
            # 整理 resolution：若都一样，返回标量；否则返回列表（与 image_paths 对齐）
            # 去掉 None 再判断是否全相等
            non_none = [r for r in resolutions if r is not None]
            if non_none and all(abs(non_none[0] - r) < 1e-12 for r in non_none):
                resolution: Any = non_none[0]
            else:
                # 对齐长度：若 Resolution 个数不足，补 None
                if len(resolutions) < len(image_paths):
                    resolutions.extend([None] * (len(image_paths) - len(resolutions)))
                resolution = resolutions
        
            return image_paths, resolution, measuredStoma
        def load_Measure():
            
            xml_file = filedialog.askopenfilename(
            title="选择测量结果 XML",
               filetypes=[("XML files", "*.xml"), ("All files", "*.*")]
            )

            if xml_file:
               img_paths, resolution, measured = load_measured_xml(xml_file)

               self.image_paths = img_paths
               self.measuredStoma = measured
               for fn in img_paths:
                   self.listboxFile.insert(tk.END, os.path.basename(fn))
            
        def save_Stoma(event=None):
            
            if len(self.__imframe.curROICoord) == 2:
                self.listboxStoma.delete(0, tk.END) 
                print(self.__imframe.curIdx,self.__imframe.measuredROI)
                if self.__imframe.curIdx < len(self.__imframe.measuredROI):
                    
                    self.__imframe.measuredROI[self.__imframe.curIdx ] = self.__imframe.curROICoord.copy()
                    self.__imframe.curIdx =len(self.__imframe.measuredROI)-1
                else:
                    self.__imframe.measuredROI.append(self.__imframe.curROICoord.copy())
                    #self.__imframe.curIdx = self.__imframe.curIdx + 1
                self.__imframe.curROICoord.clear()
                #self.curIdx = len(self.__imframe.measuredROI)
                self.measuredStoma[self.index] = self.__imframe.measuredROI
            
                #self.__imframe.count = self.__imframe.count + 1
                stoma = len(self.__imframe.measuredROI)
                for i in range(stoma):
                   self.listboxStoma.insert(tk.END, self.Num2String(i))#第{i+1}番目気孔#
        
        def save_Measure():
            
            folder_path = filedialog.askdirectory()
            
            if folder_path:
                now = datetime.now()
                formatted = now.strftime("%Y-%m-%d-%H-%M")
                resultxml = os.path.join(folder_path,f"Measured-{formatted}.xml")
                xroot = ET.Element("data")
               # 创建图像路径节点
                for idx in range(len(self.image_paths)):
                     image_node = ET.SubElement(xroot, "imagePath")
                     image_node.text = self.image_paths[idx]
                     image_Res = ET.SubElement(xroot, "Resolution")
                     image_Res.text = str(self.resolution)
               # 创建XML树
                     for i,each in enumerate(self.measuredStoma[idx]):
                        roi = ET.SubElement(xroot, "Measure")
                        roi.set("Measure", str(i+1))
                        element = ET.SubElement(roi, 'item')
                        ET.SubElement(element, 'ltx').text = str(each[0][0])
                        ET.SubElement(element, 'lty').text = str(each[0][1])
                        ET.SubElement(element, 'rbx').text = str(each[1][0])
                        ET.SubElement(element, 'rby').text = str(each[1][1])
                        
                tree = ET.ElementTree(xroot)
                # 将XML写入文件
                tree.write(resultxml, encoding="utf-8", xml_declaration=True)

        def save_image():
            
            folder_path = filedialog.askdirectory()
            
            if folder_path:
                now = datetime.now()
                formatted = now.strftime("%Y-%m-%d-%H-%M")
                resultExcel = os.path.join(folder_path,f"Measured-{formatted}.csv")
                with open(resultExcel,'a',newline='',encoding='utf-8') as f:
                    write = csv.writer(f)
                    write.writerow(["Measure Date:",f"{formatted}"])
                    write.writerow(["Image Name","Stoma ID", "Pixel Distance", f"Physical Distance ({self.unit})"])
                    
                    
                    
                    
                    for idx, image_path in enumerate(self.image_paths):
                        try:
                            ROIs = self.measuredStoma[idx]
                            
                            if ROIs is not None:
                                image = Image.open(image_path).convert("RGB")
                                draw = ImageDraw.Draw(image)
                                
                                font = ImageFont.truetype("arial.ttf", size=36)
 
                                filename = os.path.basename(image_path)
            
                                for i, roi in enumerate(ROIs):
                                    ltx =int(roi[0][0])
                                    lty =int(roi[0][1])
                                    rbx =int(roi[1][0])
                                    rby =int(roi[1][1])
                                    draw.line((ltx,lty,rbx,rby), fill="red", width=3)
                                    x,y = max(ltx,rbx),max(lty,rby)
                                    distance = math.hypot(rbx - ltx, rby - lty)
                                    draw.text((x, y), f"Stoma ID:  {i+1}", fill="blue", font=font)
                                    draw.text((x, y+30), f"Pixel:    {distance:.1f} pixel", fill="blue", font=font)
                                    draw.text((x, y+60), f"Distance: {distance*self.resolution:.1f} {self.unit}", fill="blue", font=font)
                                    write.writerow([f"{filename}",f"{i+1}",f"{distance:.1f}",f"{distance*self.resolution:.1f}"])
                                output_path = os.path.join(folder_path, filename)
                                if "png" in filename:
                                    image.save(output_path)
                                else:
                                    image.save(output_path,quality=95)

                        except Exception as e:
                            messagebox.showinfo(f"{GUI_LGE['error'][self.curlang]}",f"{GUI_LGE['errormgs'][self.curlang]} {image_path}: {e}")
                            
        top = tk.Toplevel()
        top.geometry("1600x1000")
        paned_window = tk.PanedWindow(top, orient=tk.HORIZONTAL, sashwidth=5)
        paned_window.pack(fill=tk.BOTH, expand=1)

        # 左侧固定宽度框架
        left_frame = tk.Frame(paned_window, width=150, bg="#b0c4de")
        left_frame.pack_propagate(False)  # 禁止自动缩放
        paned_window.add(left_frame)

        # 右侧可缩放框架
        right_frame = tk.Frame(paned_window)#, bg="#4682b4"
        paned_window.add(right_frame)
        
        # 左侧控件
        
        btn_folder = tk.Button(left_frame, text="Open folder", command=open_folder)#フォルダを開く
        btn_folder.pack(fill=tk.X, padx=5, pady=5)
        
        btn_Load = tk.Button(left_frame, text="Load Measure (xml)", command=load_Measure)#フォルダを開く
        btn_Load.pack(fill=tk.X, padx=5, pady=5)
        
        btn_save = tk.Button(left_frame, text="Save (csv)", command=save_image)#計測保存
        btn_save.pack(fill=tk.X, padx=5, pady=5)
        
        btn_saveXml = tk.Button(left_frame, text="Save (xml)", command=save_Measure)#計測保存
        btn_saveXml.pack(fill=tk.X, padx=5, pady=5)
        
        Folder_label = tk.Label(left_frame,text ="Image list", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#f0f0f0")#画像リスト
        Folder_label.pack(pady=(10, 5))
        
        self.listboxFile = tk.Listbox(left_frame)
        #for i in range(10):
        #    self.listbox.insert(tk.END, f"Image{i}")
        self.listboxFile.bind("<<ListboxSelect>>", self.display_selected_image)

        self.listboxFile.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        Stoma_label = tk.Label(left_frame,text ="Stoma list", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#f0f0f0")#気孔リスト
        Stoma_label.pack(pady=(10, 5))
        
        
        self.listboxStoma = tk.Listbox(left_frame)
        #for i in range(4):
        #    self.listboxStoma.insert(tk.END, f"Image{i}")
        self.listboxStoma.bind("<<ListboxSelect>>", self.display_selected_stoma)
        self.listboxStoma.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 右侧图像显示区域（示例用Label占位）
        #image_label = tk.Label(right_frame, text="計測する前に画像かフォルダを開いてください。", bg="#4682b4", fg="white")
        #image_label.pack(fill=tk.BOTH, expand=True)

        toolbar = tk.Frame(right_frame)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        def set_image_mode(mode):
            if hasattr(self, "__imframe") and self.__imframe is not None:
                self.__imframe.set_mode(mode)
        
        tk.Button(toolbar, text="Measure", command=lambda: set_image_mode("measure")).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Brush +", command=lambda: set_image_mode("brush_fg")).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Brush -", command=lambda: set_image_mode("brush_bg")).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Raw", command=lambda: self.switch_display_mode("raw")).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Overlay", command=lambda: self.switch_display_mode("overlay")).pack(side=tk.LEFT, padx=2)
        
        self.__placeholder = tk.Frame(right_frame, borderwidth=1, relief="solid")
        self.__placeholder.pack(fill=tk.BOTH, expand=True)
        self.__placeholder.rowconfigure(0, weight=1)
        self.__placeholder.columnconfigure(0, weight=1)
        
        self.__imframe = ImageFrame(placeholder=self.__placeholder, roi_size= (1600,1200),curImg = None,status_var = self.status_var, resolu = self.resolution, unit = self.unit, Pixel_var = self.Pixel_var)

        
        # 固定左侧宽度
        def on_resize(event):
            left_frame.config(width=150)
        
        left_frame.bind("<Configure>", on_resize)
        top.bind("<Control-s>", save_Stoma)
        
    def SetResolution(self):
        #self.GenerateReport()
        self.scaleWindow = tk.Toplevel(self.root)
        self.scaleWindow.title(GUI_LGE['scalesetting'][self.curlang])
        #self.scaleWindow.iconbitmap("root.ico") 
        self.scaleWindow.geometry("{}x{}".format(300,200))
        labelPhy = tk.Label(self.scaleWindow, text = GUI_LGE['knowndistance'][self.curlang])
        self.txtPhy = tk.Entry(self.scaleWindow,width=20)
        self.txtPhy.insert(0,"20")
        labelPixel = tk.Label(self.scaleWindow, text = GUI_LGE['knownpixel'][self.curlang])
        self.txtPixel = tk.Entry(self.scaleWindow,width=20,textvariable= self.Pixel_var)
        self.txtPixel.delete(0, tk.END)	
        self.txtPixel.insert(0,"130")
        labelUnit = tk.Label(self.scaleWindow, text = GUI_LGE['knownunit'][self.curlang])
        #v = tk.StringVar()
        #v.set("um")
        self.cb = ttk.Combobox(self.scaleWindow, values=['cm',"um",'mm'], width=10)#textvariable=v, 
        self.cb.current(1)
        self.cb.grid(row=2, column=1)
        
        
        buttonOpenImg = tk.Button(self.scaleWindow, text = GUI_LGE['knownimage'][self.curlang],command=self.OpenImage)
        buttonOK = tk.Button(self.scaleWindow, text = "OK",command=self.__ScaleSetted)
        
        labelPhy.grid(row=0, column=0)
        labelPixel.grid(row=1, column=0)
        labelUnit.grid(row=2, column=0)
        self.txtPhy.grid(row=0, column=1)
        self.txtPixel.grid(row=1, column=1)
        
        buttonOpenImg.grid(row=3, column=0)
        buttonOK.grid(row=3, column=1)
        
        self.scaleWindow.resizable(False, False)
        
    def open_settings(self):
        #top = tk.Toplevel(self.root)
        #top.title("设置")
        #tk.Label(top, text="此处为设置界面", padx=20, pady=20).pack()
        self.SetResolution()
        self.update_status(GUI_LGE['statussetting'][self.curlang])

    def show_help(self):
        top = tk.Toplevel(self.root)
        top.title("Help")#ヘルプ
        tk.Label(top, text=GUI_LGE['help'][self.curlang], padx=20, pady=20).pack()
        self.update_status(GUI_LGE['helptitle'][self.curlang])#ヘルプを開きました。

    def update_status(self, text):
        self.status.set(text)

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartApp(root)
    root.mainloop()
