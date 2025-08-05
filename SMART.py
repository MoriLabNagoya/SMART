import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import  messagebox
from PIL import Image
import glob
import os
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
#from tkinter import PhotoImage

language_order = ["en", "ja", "zh"]

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
        self.extensions = ['*.jpg', '*.tif']
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, r"resource\last-ResSeg-trans.hdf5")
        yolo_path = os.path.join(base_dir, r"resource\best.pt")
        self.yolo_model = YOLO(yolo_path)  # load a custom 
        self.segmodel = resnet50_segnet(n_classes=4, input_height=256, input_width=256,btrans = True)
        self.segmodel.load_weights(model_path)# by_name=False
        
    def load_icons(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.root.iconbitmap(os.path.join(base_dir, r"resource\smart.ico"))
        #img = PhotoImage(file=r"resource\smart.png")
        #self.root.iconphoto(True, img)
        #print("")

    def create_top_buttons(self):
        frame = tk.Frame(self.root, bg="lightblue")
        frame.pack(fill=tk.X)
        

        tk.Button(frame, text="+", font=self.big_font, width=2, height=1, command=self.add_row).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="-", font=self.big_font, width=2, height=1, command=self.remove_row).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text=">", font=self.big_font, width=2, height=1, command=self.StartProcess).pack(side=tk.LEFT, padx=5)
        #tk.Button(frame, text="1", font=self.big_font, width=2, height=1, command=self.Process).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="S", font=self.big_font, width=2, height=1, command=self.open_settings).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="C", font=self.big_font, width=2, height=1, command=self.Clear).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="M", font=self.big_font, width=2, height=1, command=self.measurement).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="L", font=self.big_font, width=2, height=1, command=self.ChangeLanguage).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="?", font=self.big_font, width=2, height=1, command=self.show_help, bg="lightgray").pack(side=tk.RIGHT, padx=5)
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
        self.checkbox_text.set("Process all stomata")
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
        path_var.set(r"C:\Users\chwang\Desktop\result-leaf")
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
    def ProcessROI(self,curImage, bb, idx, angle, resolution,c,compound,fname):
        One_Stat = []
        x1,y1,x2,y2 = bb[idx].astype(np.int32)
        if (y2-y1)*(x2-x1)>0 and x1>0 and y1>0:
              roi = curImage[y1:y2,x1:x2]
              #cv2.imwrite(os.path.join(resultPath,os.path.basename(fname)+f"{idx+1}.jpg"), roi.copy())
              
              out = self.segmodel.predict_segmentation(roi)
              guard = out==2
              apert = out==1
              mouse = out==3
              guard = cv2.resize(guard.astype(np.uint8), (roi.shape[1],roi.shape[0]),cv2.INTER_CUBIC)
              #guard = largest_region(guard)
              apert = cv2.resize(apert.astype(np.uint8), (roi.shape[1],roi.shape[0]),cv2.INTER_CUBIC)
              apert = self.keep_largest_component(apert)
              #apert = largest_region(apert)
              mouse = cv2.resize(mouse.astype(np.uint8), (roi.shape[1],roi.shape[0]),cv2.INTER_CUBIC)
              gw, gh, ga = self.AnalyseMorphy(guard, angle)
              aw, ah, aa = self.AnalyseMorphy(apert, angle)
              ma = np.sum(mouse)
              out = guard*2 + apert + mouse*3
              roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
              for one in range(1,3):
                  roi[(out==one)] = self.COLORS[one%len(self.COLORS)]
                  curImage[y1:y2,x1:x2] = roi
              #rotated_out = rotate_image(roi, -angle*180/3.1415)
              
             
              #cv2.rectangle(curImage, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[c], 2) 
              od = 0
              if ah>0:
                 od = aw/ah#aw*gh / (ah*gw)f"{num:.2f}"
              textx,texty = int(x1), int(y1)-10
              if x1 < 50:
                  textx = int(x2)
              if y1  < 50:
                  texty = int(y2) + 10
              #if x2 >curImage.shape[0] -50:
              if ah ==0:
                  #cv2.putText(curImage, f'od {idx+1}: 0 %', (textx,texty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                  cv2.putText(curImage, f'Stoma ID {idx+1}: 0 %', (textx,texty+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
              else:
                  #cv2.putText(curImage, f'od {idx+1}: {100*aw/ah:.2f} %', (textx,texty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                  cv2.putText(curImage, f'Stoma ID: {idx+1}: {100*4*aa/(ah*ah*3.1416):.2f} %', (textx,texty+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
              
              
              #cv2.putText(curImage, f'{aw}: {ANOTATE[c]},{od*100:.2f} %', (textx,texty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
              
              One_Stat = [gw*resolution,gh*resolution,ga*resolution*resolution,aw*resolution,ah*resolution,aa*resolution*resolution,ma*resolution*resolution,od,compound,idx+1,self.ANOTATE[c],fname]        
        return One_Stat
    def Process_OneImage(self,fname,resultPath,resolution,compound):
        StatisticsROIs = []
        with open(fname, 'rb') as f:
                image_data = f.read()
                curImage = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                result = self.yolo_model(curImage)
                His = list(result[0].obb.xywhr.numpy())
                clss = list(result[0].obb.cls.numpy().astype(np.int8))
                bb = list(result[0].obb.xyxy.numpy())
                
                
                # 处理和可视化最终结果（可选）
                #cnt = 0
                for idx,c in enumerate(clss):
                    cx,cy,w,h,angle = His[idx]
                    if self.checkbox_var.get():
                        Statistics= self.ProcessROI(curImage,bb, idx,angle,resolution,c,compound,os.path.basename(fname))
                        StatisticsROIs.append(Statistics)
                    elif c==0:
                        Statistics = self.ProcessROI(curImage,bb, idx,angle,resolution,c,compound,os.path.basename(fname))
                        StatisticsROIs.append(Statistics)
                        #cnt = cnt + 1
                        #rect = ((cx, cy), (w, h), angle*180/3.1415)  # ((中心点), (宽, 高), 旋转角度)
                        #box = cv2.boxPoints(rect)  # 获取四个角点
                        #box = np.int32(box)  # 转换为整数
                        # 画旋转框
                        #cv2.polylines(curImage, [box], isClosed=True, color=COLORS[c], thickness=2)
                        
                                  
                #file_count = len(glob.glob(os.path.join(resultPath, "*")))  # 统计所有文件和子文件夹
        #cv2.imwrite(os.path.join(resultPath,compound, os.path.basename(fname)), curImage)
        self.save_image_with_increment(os.path.join(resultPath,compound, os.path.basename(fname)), curImage)
        self.progress["value"] += 1
        self.root.update_idletasks()  # 更新进度条显示
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
    def summarize_with_outlier_filtering(self,datas,compoundList):
        """
        输入：
            d : list 或 numpy array，表示一组数值
        输出：
            dict，包含原始值和去除离群值后的 max, min, mean
        """
        sums =list()
        for i,d in enumerate(datas):
            d = np.array(d)
            summary = {}
            # 原始数据统计
            summary['Compound'] = compoundList[i]
            summary['raw_max'] = float(np.max(d))
            summary['raw_min'] = float(np.min(d))
            summary['raw_mean'] = float(np.mean(d))

            # 使用 IQR 过滤离群值
            Q1 = np.percentile(d, 25)
            Q3 = np.percentile(d, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            filtered = d[(d >= lower_bound) & (d <= upper_bound)]

        # 如果过滤后为空，设置为 None
            if len(filtered) > 0:
                summary['filtered_max'] = float(np.max(filtered))
                summary['filtered_min'] = float(np.min(filtered))
                summary['filtered_mean'] = float(np.mean(filtered))
            else:
                summary['filtered_max'] = None
                summary['filtered_min'] = None
                summary['filtered_mean'] = None
            sums.append(summary)
        return sums
    def DataStatis(self,AllResults):
        awList = []
        poreSizeList = []
        StomaSizeList = []
        
        compoundList = []
        for each in AllResults:
              each = [x for x in each if x]
              aw = [item[3] for item in each if len(item) > 0]
              awList.append(aw)
              #od = [item[7] for item in each if len(item) > 0]
              #odList.append(od)
         
              pore = [item[5] for item in each if len(item) > 0]
              poreSizeList.append(pore)
              
              stoma = [item[2]+item[5]+item[6] for item in each if len(item) > 0]
              
              StomaSizeList.append(stoma)
         
              tmp = [item[8] for item in each if len(item) > 0]
              compound = list(set(tmp))
              compoundList.append(compound)   
        return self.summarize_with_outlier_filtering(awList,compoundList),self.summarize_with_outlier_filtering(poreSizeList,compoundList),self.summarize_with_outlier_filtering(StomaSizeList,compoundList)
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
            #write.writerow(["Resolution:",f"{resolution} {self.unit}"])
            aw,pore, stoma = self.DataStatis(AllResults)
            write.writerow(["Compound", "Item","Max (all)","Min (all)","Mean (all)","Max (IQR)","Min (IQR)","Mean (IQR)"])
            for one in aw:
                write.writerow([f"{one['Compound']}","Pore width",f"{one['raw_max']:.2f}",f"{one['raw_min']:.2f}",f"{one['raw_mean']:.2f}",f"{one['filtered_max']:.2f}",f"{one['filtered_min']:.2f}",f"{one['filtered_mean']:.2f}"]) 
            for one in pore:
                write.writerow([f"{one['Compound']}","Pore size",f"{one['raw_max']:.2f}",f"{one['raw_min']:.2f}",f"{one['raw_mean']:.2f}",f"{one['filtered_max']:.2f}",f"{one['filtered_min']:.2f}",f"{one['filtered_mean']:.2f}"]) 
            for one in stoma:
                write.writerow([f"{one['Compound']}","Stoma size",f"{one['raw_max']:.2f}",f"{one['raw_min']:.2f}",f"{one['raw_mean']:.2f}",f"{one['filtered_max']:.2f}",f"{one['filtered_min']:.2f}",f"{one['filtered_mean']:.2f}"]) 
            write.writerow(["FileName","Stoma ID","Type",f"Guard Width ({self.unit})",f"Guard Height ({self.unit})",f"Guard Size ({self.unit}*{self.unit})",f"Aperture Width ({self.unit})",f"Aperture Height ({self.unit})",f"Aperture Size ({self.unit}*{self.unit})",f"Mouse ({self.unit}*{self.unit})","Opening Degree","Compound Name"])
            for one in AllResults:
                for stat in one:
                    write.writerow([stat[11],stat[9],stat[10],stat[0],stat[1],stat[2],stat[3],stat[4],stat[5],stat[6],stat[7],stat[8]])                        
        return AllResults
            #(awList, "Stomata width (um)", "Stomata width", os.path.join(savefolder,"compare-pore-width"),compoundList)
    def GenerateFigure(self, dataList, ylabel_text, title, savePath, compoundList, isPore):
        #import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if self.lang_index ==0:
            myfont = fm.FontProperties(family='Arial')  
        elif self.lang_index ==1:
            myfont = fm.FontProperties(fname=os.path.join(base_dir, r"resource\msgothic.ttc"))
        else:
            myfont = fm.FontProperties(fname=os.path.join(base_dir, r"resource\simhei.ttf"))

            #fontPath = ['','msgothic.ttc','simhei.ttf']
        
        from scipy.stats import f_oneway
        #from statsmodels.stats.multicomp import pairwise_tukeyhsd
        boxprops = dict(linewidth=1)
        whiskerprops = dict(linewidth=1)
        capprops = dict(linewidth=1)
        medianprops = dict(linewidth=1)
        flierprops = dict(marker='o', markerfacecolor='red', markersize=2, markeredgecolor='black')

        positions = list(range(1, len(compoundList) + 1))  # 位置 [1, 2, 3...]

        plt.figure(figsize=(10, 10))

    # 箱线图
        bp = plt.boxplot(
            dataList,
            positions=positions,
            tick_labels=compoundList,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops,
            flierprops=flierprops
        )
        plt.xticks(fontsize=20)   # x轴刻度字体大小
        plt.yticks(fontsize=20) 

    # 添加散点图
        for i, d in enumerate(dataList):
            x = np.random.normal(loc=positions[i], scale=0.05, size=len(d))
            plt.scatter(x, d, alpha=0.7, s=40)

        # 添加每组的 n = 数字 标签（在箱体上方）
        for i, d in enumerate(dataList):
            max_val = np.max(d)
            plt.text(
                positions[i],
                max_val + (0.01 * max_val),
                f"n = {len(d)}",
                ha='center',
                va='bottom',
                fontsize=9,
                color='black'
            )
        if len(compoundList)>1:
            from scipy.stats import tukey_hsd       
            anova_result = f_oneway(*dataList)
            summary_data = tukey_hsd(*dataList)
            anova_text = f"ANOVA: F={anova_result.statistic:.2f}, p={anova_result.pvalue:.3e}"
            plt.gcf().text(0.1, 0.95, anova_text, fontsize=20, ha='left')

            #summary_data = tukeyhsd_result.summary().data[1:]
            y_max = max([max(g) for g in dataList])
        
            y_offset = (y_max * 0.05) if y_max > 0 else 1
            current_height = y_max + y_offset

            def p_to_star(p):
                if p < 0.001:
                    return '***'
                elif p < 0.01:
                    return '**'
                elif p < 0.05:
                    return '*'
                else:
                    return ''

            pvalues = summary_data.pvalue  # shape: (n_groups, n_groups)
           # group_names = compoundList  # 所有组名，等价于 compoundList 顺序
            ci = summary_data.confidence_interval()
            ci_low = ci.low
            ci_high = ci.high
            summary_rows = []
            stats = summary_data.statistic
            
            n = len(compoundList)
            for i in range(n):
                for j in range(i + 1, n):
                    g1 = compoundList[i]
                    g2 = compoundList[j]
                    meandiff = stats[i, j]
                    p_adj = pvalues[i, j]
                    lower = ci_low[i, j]
                    upper = ci_high[i, j]
                    reject = p_adj < 0.05  # 显著性判
                    summary_rows.append((g1, g2, meandiff, p_adj, lower, upper, reject))

            line_count = 0
            # 绘图（和你原来的逻辑一致）
            for res in summary_rows:
                g1, g2, meandiff, p_adj, lower, upper, reject = res
                star = p_to_star(p_adj)
                if star:
                    i = compoundList.index(g1)
                    j = compoundList.index(g2)
                    x1, x2 = positions[i], positions[j]
                    y = current_height + (line_count * y_offset * 0.7)
                    # 画连接线
                    plt.plot([x1, x1, x2, x2], [y, y+0.2, y+0.2, y], lw=1.5, c='black')
                    plt.text((x1 + x2) / 2, y + 0.25, star, ha='center', va='bottom', color='black', fontsize=20)
                    plt.text((x1 + x2) / 2, y - 0.45, f"p={p_adj:.2e}", ha='center', va='bottom', color='black', fontsize=20)
                    line_count += 1

        # 标签与保存
        plt.ylim(bottom=0)
        plt.ylabel(ylabel_text,fontproperties=myfont,fontsize=20)
        plt.title(title,fontproperties=myfont,fontsize=20)
        plt.tight_layout()
        plt.savefig(savePath)
        plt.close()

    def GenerateCompare(self,AllResults,savefolder):#StomaStat_1,StomaStat_2, savefolder, pre_com, cur_com
       awList = []
       odList = []
       poreSizeList = []
       StomaSizeList = []
       
       compoundList = []
       for each in AllResults:
            each = [x for x in each if x]
            aw = [item[3] for item in each if len(item) > 0]
            awList.append(aw)
            od = [item[7] for item in each if len(item) > 0]
            odList.append(od)
            
            pore = [item[5] for item in each if len(item) > 0]
            poreSizeList.append(pore)
            
            stoma = [item[2]+item[5]+item[6] for item in each if len(item) > 0]
            
            StomaSizeList.append(stoma)
            
            tmp = [item[8] for item in each if len(item) > 0]
            compound = list(set(tmp))
            compoundList.append(compound)  
            
        #aw_1 = [item[0] for item in StomaStat_1 if len(item) > 0]
        #aw_2 = [item[0] for item in StomaStat_2 if len(item) > 0]
       self.GenerateFigure(awList, f"{GUI_LGE['porewidth'][self.curlang]} ({self.unit})", f"{GUI_LGE['porewidth'][self.curlang]}", os.path.join(savefolder,"compare-pore-width.png"),compoundList,True)
       self.GenerateFigure(poreSizeList, f"{GUI_LGE['porearea'][self.curlang]} ({self.unit}*{self.unit})", f"{GUI_LGE['porearea'][self.curlang]}", os.path.join(savefolder,"compare-pore-area.png"),compoundList,False)
       self.GenerateFigure(StomaSizeList, f"{GUI_LGE['stomaarea'][self.curlang]} ({self.unit}*{self.unit})", f"{GUI_LGE['stomaarea'][self.curlang]}", os.path.join(savefolder,"compare-stoma-size.png"),compoundList,False)
       
       awA, awB,aA, aB,rA,rB = list(),list(),list(),list(),list(),list()
       for each in AllResults:
           each = [x for x in each]
           awA = [item[3] for item in each if len(item) > 0 and item[8]==compoundList[0][0]]
           awB = [item[3] for item in each if len(item) > 0 and item[8]==compoundList[1][0]]
           
           aA = [item[5] for item in each if len(item) > 0 and item[8]==compoundList[0][0]]
           aB = [item[5] for item in each if len(item) > 0 and item[8]==compoundList[1][0]]
           
           rA = [item[7] for item in each if len(item) > 0 and item[8]==compoundList[0][0]]
           rB = [item[7] for item in each if len(item) > 0 and item[8]==compoundList[1][0]]
       awA = sum(awList[0]) / len(awList[0])
       awB = sum(awList[1]) / len(awList[1])
       aA = sum(poreSizeList[0]) / len(poreSizeList[0])
       aB = sum(poreSizeList[1]) / len(poreSizeList[1])
       rA = sum(odList[0]) / len(odList[0])
       rB = sum(odList[0]) / len(odList[0])
       self.GenerateReport(compoundList[0][0],compoundList[1][0],awA, awB,aA, aB,rA,rB)
    def GenerateReport(self,compoundAName, compoundBName, 
                   aperturewidthA, aperturewidthB, 
                   areaA, areaB, 
                   ratioA, ratioB, 
                   filename="stomatal_report.txt"):

        prompt = f"""You are shown an image or chart comparing stomatal parameters between two compounds.  
Please analyze and interpret the differences between **{compoundAName}** and **{compoundBName}**, based on the following quantitative values:

- **Mean stomatal aperture width** (in μm)  
- **Mean stomatal aperture area** (in μm²)  
- **Mean width/length ratio** (dimensionless)

The data are:  
- **{compoundAName}**:  
  - Width: {aperturewidthA:.2f} μm  
  - Area: {areaA:.2f} μm²  
  - Width/length ratio: {ratioA:.2f}  

- **{compoundBName}**:  
  - Width: {aperturewidthB:.2f} μm  
  - Area: {areaB:.2f} μm²  
  - Width/length ratio: {ratioB:.2f}  
  Please provide a scientific interpretation of how Compound **{compoundBName}** affects stomatal opening compared to **{compoundAName}**
"""

    # 写入（或追加）到文本文件
        collected = []    
        for _, path_var, extra_entry in self.rows:
            collected.append({
                "path": path_var.get()
            })
            
        with open(filename, "a", encoding="utf-8") as f:
            f.write(str(collected))
            f.write(prompt)

        
        
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
    def Process(self):
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
        if os.path.exists(os.path.join(resultFolder,"tmp")) ==False:
            os.makedirs(os.path.join(resultFolder,"tmp"))
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
        for item in collected:#process each compound
            raw_path_string = item["path"] #one item is one compound
            compound = item["input"]
            target = os.path.join(resultFolder,"tmp",compound)
            if os.path.exists(target) ==False:
               os.mkdir(target)
            path_list = [p.strip() for p in raw_path_string.split(",") if p.strip()]
            item["path"] =target
            imgs = []
        
            for path in path_list:
               # 对每个单独路径进行处理
               for ext in self.extensions:
                   # 使用 glob 查找匹配的文件
                   imgs.extend(glob.glob(os.path.join(path, ext)))
            for fname in imgs:
                with open(fname, 'rb') as f:
                    image_data = f.read()
                    curImage = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                    cv2.resize(curImage, (curImage.shape[0]//4,curImage.shape[1]//4))
                    bg = np.zeros([2000,2000,3],np.uint8)
                    bg[500:500+curImage.shape[0],500:500+curImage.shape[1]] = curImage
                    self.save_image_with_increment(os.path.join(target,os.path.basename(fname)), bg)

        AllResults = self.ProcessImages(collected,resultFolder,float(self.resolution))
        self.progress["value"] = 100  # 假设完成
        self.update_status(f"{GUI_LGE['finished'][self.curlang]}")
        self.GenerateCompare(AllResults,resultFolder)
        self.update_status(f"{GUI_LGE['saved'][self.curlang]}")
        
        messagebox.showinfo(f"{GUI_LGE['finish'][self.curlang]}",f"{GUI_LGE['finished'][self.curlang]}")#完成","気孔の計測が完了しました。
    def StartProcess(self):
        
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
        
        AllResults = self.ProcessImages(collected,resultFolder,float(self.resolution))
        self.progress["value"] = 100  # 假设完成
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
            self.indexStoma = selection[0]
            #self.__imframe.ROICoord = self.measuredStoma[self.index][self.indexStoma].copy()        
            #self.__imframe.ReFresh(event)
            
    def display_selected_image(self, event):
        selection = self.listboxFile.curselection()
        if selection:
            if len(self.__imframe.ROICoord)==2:
                self.__imframe.measuredROI.append(self.__imframe.ROICoord)
                self.measuredStoma[self.index] = self.__imframe.measuredROI.copy()
            
            self.index = selection[0]
            filepath = self.image_paths[self.index]

            
            image = Image.open(filepath)

            self.__imframe = ImageFrame(placeholder=self.__placeholder, roi_size= image.size,curImg = image,status_var = self.status_var, resolu = self.resolution, unit = self.unit, Pixel_var = self.Pixel_var)
            
            self.__imframe.SetROI(self.measuredStoma[self.index])
            self.__imframe.ReFresh(event=None)
            
            self.listboxStoma.delete(0, tk.END)
            stoma = len(self.__imframe.measuredROI)
            for i in range(stoma):
               self.listboxStoma.insert(tk.END, f"The {i+1}-th stoma")#第{i+1}番目気孔
            
    def clear_list(self):
        self.listboxFile.delete(0, tk.END)     
        self.listboxStoma.delete(0, tk.END)    
        self.image_paths.clear()            
        
    def measurement(self):
        if self.resolution ==0:
            messagebox.showinfo(GUI_LGE['warning'][self.curlang],GUI_LGE['resolutionset'][self.curlang])#警告","解像度を設定してください！
            return
        self.image_paths = []
        self.measuredStoma = []
        def open_image():
            
            self.clear_list()
            filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
            if filepath:
                self.image_paths.append(filepath)
                
                self.listboxFile.insert(tk.END, os.path.basename(filepath))
                self.measuredStoma = [None]
                self.index = 0
                image = Image.open(filepath)
                self.__imframe = ImageFrame(placeholder=self.__placeholder, roi_size= image.size,curImg = image,status_var = self.status_var, resolu = self.resolution, unit = self.unit, Pixel_var = self.Pixel_var)
        def open_folder():
            
            folder_path = filedialog.askdirectory()
            self.clear_list()
            if folder_path:
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        full_path = os.path.join(folder_path, filename)
                        self.image_paths.append(full_path)
                        self.listboxFile.insert(tk.END, filename)
                self.measuredStoma = [None] * len(self.image_paths)
        def save_Stoma(event=None):
            
            if len(self.__imframe.ROICoord) == 2:
                self.listboxStoma.delete(0, tk.END) 
                self.__imframe.measuredROI.append(self.__imframe.ROICoord.copy())
                self.__imframe.ROICoord.clear()

                self.measuredStoma[self.index] = self.__imframe.measuredROI
            
                #self.__imframe.count = self.__imframe.count + 1
                stoma = len(self.__imframe.measuredROI)
                for i in range(stoma):
                   self.listboxStoma.insert(tk.END, f"The {i+1}-th stoma")#第{i+1}番目気孔#
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
        btn_image = tk.Button(left_frame, text="Open image", command=open_image)#画像を開く
        btn_image.pack(fill=tk.X, padx=5, pady=5)
        
        btn_folder = tk.Button(left_frame, text="Open folder", command=open_folder)#フォルダを開く
        btn_folder.pack(fill=tk.X, padx=5, pady=5)
        
        btn_save = tk.Button(left_frame, text="Save measurement", command=save_image)#計測保存
        btn_save.pack(fill=tk.X, padx=5, pady=5)
        
        
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

        self.__placeholder = tk.Frame(right_frame,borderwidth=1, relief="solid")#, bg="#4682b4"
        self.__placeholder.grid(row=0, column=0, sticky='nswe')
        self.__placeholder.rowconfigure(0, weight=1)  # make grid cell expandable
        self.__placeholder.columnconfigure(0, weight=1)
        self.__placeholder.pack(fill=tk.BOTH, expand=True)
        self.__imframe = ImageFrame(placeholder=self.__placeholder, roi_size= (1600,1200),curImg = None,status_var = self.status_var, resolu = self.resolution, unit = self.unit, Pixel_var = self.Pixel_var)

        
        # 固定左侧宽度
        def on_resize(event):
            left_frame.config(width=150)
        
        left_frame.bind("<Configure>", on_resize)
        top.bind("<Control-s>", save_Stoma)
        top.mainloop()
        
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
        #self.OpenImage()
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
