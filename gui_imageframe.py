# -*- coding: utf-8 -*-
import tkinter as tk

from tkinter import ttk
from PIL import Image, ImageTk
#from .logic_logger import logging
from gui_autoscrollbar import AutoScrollbar
import math
import copy
import numpy as np
import cv2
class ImageFrame():
    """ Display an image and necessary functional for rectangle, zoom, shift, etc. """
    #def __init__(self, placeholder, path, roi_size,curImg):
    def __init__(self, placeholder, roi_size,curImg,status_var,resolu, unit, Pixel_var):#PIL image
        """ Initialize the ImageFrame """
        #self.path = path  # path to the image, should be public to remember it into INI config file
        self.__roi_size = roi_size  # obtain size of the roi
        self.__roi_tag = 'roi'
        self.__text_size = 14  # size of the text
        self.__text_tag = 'text'
        self.__font = 'Helvetica {size} normal'
        self.__font_max_size = 20  # max font size
        self.__font_min_size = 7  # min font size
        self.__imscale = 1.0  # scale for the canvas image zoom
        self.__delta = 1.3  # zoom magnitude
        self.__previous_state = 0  # previous state of the keyboard
        self.Pixel_Distance = Pixel_var

        self.__imframe = tk.Frame(placeholder,borderwidth=1, relief="solid")  # placeholder of the ImageFrame object
        self.__imframe.grid(row=0, column=0, sticky='nsew',padx=5, pady=5)
        self.__imframe.rowconfigure(0, weight=1)  # make grid cell expandable
        self.__imframe.columnconfigure(0, weight=1)
        self.status_var = status_var
        self.resolution = resolu
        self.unit = unit
        #width = self.__imframe.winfo_width()
        #height = self.__imframe.winfo_height()
        #print(f"Placeholder 大小: {width}x{height}")
        
        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        # Create canvas and bind it with scrollbars
        self.__canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                  xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.__canvas.grid(row=0, column=0, sticky='nswe')
        self.__canvas.update()  # wait till canvas is created
        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)
        # Bind events to the Canvas
        self.__canvas.bind('<Configure>', self.__show_image)  # canvas is resized
        self.__canvas.bind('<ButtonPress-1>', self.__on_left_press)
        self.__canvas.bind('<B1-Motion>', self.__on_left_drag)
        self.__canvas.bind('<ButtonRelease-1>', self.__on_left_release)
        self.__canvas.bind('<ButtonPress-3>', self.__Clear)
        
        #self.__canvas.bind('<B1-Motion>', self.__move_to)  # move canvas to the new position
        self.__canvas.bind('<Motion>', self.__motion)  # handle mouse motion
        self.__canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.__canvas.bind('<Button-5>', self.__wheel)  # zoom for Linux, wheel scroll down
        self.__canvas.bind('<Button-4>', self.__wheel)  # zoom for Linux, wheel scroll up
        self.__canvas.bind('<Leave>', self.__rid_focus)  # hide roi and remove focus from the canvas
        self.__canvas.bind('<Enter>', self.__set_focus)  # set focus on the canvas
        #self.__canvas.bind("<Control-s>", self.save_action)

        # Handle keystrokes in idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time
        self.__canvas.bind('<Key>', lambda event: self.__canvas.after_idle(self.__keystroke, event))
        
        self.__state = 'hidden'
        #self.__image = Image.open(self.path)  # open image
        self.__image = curImg
        #self.__imageCV = np.array(curImg, dtype=np.uint8)
        #if curImg is not None:
        #    self.__image = curImg
        #else:
        #    self.__image = Image.open(self.path)  # open image
        # Put image into container rectangle and use it to set proper coordinates to the image
        if self.__image:
            self.__container = self.__canvas.create_rectangle((0, 0, self.__image.size), width=0)
            self.__w, self.__h = self.__image.size  # image width and height
            self.__min_side = min(self.__w, self.__h)  # get the smaller image side
        # Set region of interest (roi) rectangle on the canvas and make it invisible
        self.__roi_rect = self.__canvas.create_rectangle((0, 0, 0, 0), width=2, outline='red',
                                                         state='hidden')
        # Set text-warning on the canvas and make it invisible
        self.__text_warning = self.__canvas.create_text(
            (0, 0), anchor='sw', fill='red', text='Too close to the edge', state='hidden',
            font=self.__font.format(size=self.__text_size), tag=self.__text_tag)
        #
        
        
        self.__filter = Image.Resampling.LANCZOS  # could be: NEAREST, BILINEAR, BICUBIC and ANTIALIAS
        self.curROICoord = list()
        self.measuredROI = list()
        self.DrawList = list()
        self.scale = 1
        self.curIdx =-1
        self.mode = "measure"          # measure / brush_fg / brush_bg / view
        self.display_mode = "raw"      # raw / overlay / mask
        self.mask = None               # uint8 [H, W], 0~255
        self.brush_size = 8
        self.overlay_alpha = 0.4
        self.brush_color_value = 255   # 前景
        self._dragging = False
        if self.__image is not None:
            self.__base_image = self.__image.copy().convert("RGB")
            self.__display_image = self.__base_image.copy()
        else:
            self.__base_image = None
            self.__display_image = None
        self.__image_id = None
        if self.__image:
            self.__show_image()  # show image on the canvas
        self.__canvas.event_generate('<Enter>')  # set focus on the canvas
    def __on_left_press(self, event):
        if self.mode == "measure":
            self.__move_from(event)
            return
    
        if self.mode in ("brush_fg", "brush_bg"):
            self._dragging = True
            self.__paint_at(event)

    def __on_left_drag(self, event):
        if self.mode in ("brush_fg", "brush_bg") and self._dragging:
            self.__paint_at(event)
    
    def __on_left_release(self, event):
        self._dragging = False
    def __paint_at(self, event):
        if self.mask is None:
            h, w = self.__base_image.size[1], self.__base_image.size[0]
            self.mask = np.zeros((h, w), dtype=np.uint8)
    
        x = self.__canvas.canvasx(event.x)
        y = self.__canvas.canvasy(event.y)
        bbox1 = self.__canvas.coords(self.__container)
    
        xx = int((x - bbox1[0]) / self.__imscale)
        yy = int((y - bbox1[1]) / self.__imscale)
    
        if xx < 0 or yy < 0 or xx >= self.mask.shape[1] or yy >= self.mask.shape[0]:
            return
    
        value = 255 if self.mode == "brush_fg" else 0
        cv2.circle(self.mask, (xx, yy), self.brush_size, value, -1)
        self.display_mode = "overlay"
        self.__update_display_image()
        self.__show_image()
    def __Clear(self, event):
        if len(self.curROICoord)>0:
            self.curROICoord.pop()
            self.__show_image()
    def __move_from(self, event):
        if self.__image is None:
            return

        x = self.__canvas.canvasx(event.x)
        y = self.__canvas.canvasy(event.y)
        bbox1 = self.__canvas.coords(self.__container)
        xx = (x - bbox1[0]) / self.__imscale
        yy = (y - bbox1[1]) / self.__imscale

        # A completed current ROI starts a fresh two-click measurement.  When a
        # historical ROI is selected, keep its index so the two new points edit
        # that record; in normal add mode curIdx equals len(measuredROI).
        if len(self.curROICoord) >= 2:
            self.curROICoord = []

        if len(self.curROICoord) == 0 and not (0 <= self.curIdx <= len(self.measuredROI)):
            self.curIdx = len(self.measuredROI)

        self.curROICoord.append((xx, yy))
        self.__show_image()

    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        #self.__canvas.scan_dragto(event.x, event.y, gain=1)
        #x = self.__canvas.canvasx(event.x)
        #y = self.__canvas.canvasy(event.y)
        #print(x,y)
        self.__show_image()  # zoom tile and show it on the canvas

    def __motion(self, event):
        """ Track mouse position over the canvas """
        if self.__image == None:
            return
        x = self.__canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.__canvas.canvasy(event.y)
        w = (self.__roi_size[0] * self.__imscale) / 2
        h = (self.__roi_size[1] * self.__imscale) / 2
        bbox = self.__canvas.coords(self.__container)  # get image area
        self.status_var.set(f"{x},{y}") 
        # Draw roi rectangle
        if bbox[0] + w <= x < bbox[2] - w and bbox[1] + h <= y < bbox[3] - h:
            self.__canvas.coords(self.__roi_rect, (x - w, y - h, x + w, y + h))  # relocate roi
            if self.__state == 'hidden':
                self.__state = 'normal'
                self.__canvas.itemconfigure(self.__roi_rect, state='normal')  # show roi
                self.__canvas.itemconfigure(self.__text_warning, state='hidden')  # hide warning
        else:  # otherwise show warning
            self.__canvas.coords(self.__text_warning, (x, y))  # relocate text
            if self.__state == 'normal':
                self.__state = 'hidden'
                self.__canvas.itemconfigure(self.__text_warning, state='normal')  # show warning
                self.__canvas.itemconfigure(self.__roi_rect, state='hidden')  # hide roi
        self.__get_roi()  # update roi position in the console

    def __get_roi(self):
        """ Obtain roi image rectangle and output in the console
            upper left and bottom right corners of the rectangle """
        if self.__canvas.itemcget(self.__roi_rect, 'state') == 'normal':  # roi is not hidden
            bbox1 = self.__canvas.coords(self.__container)  # get image area
            bbox2 = self.__canvas.coords(self.__roi_rect)  # get roi area
            x1 = int((bbox2[0] - bbox1[0]) / self.__imscale)  # get upper left corner (x1,y1)
            y1 = int((bbox2[1] - bbox1[1]) / self.__imscale)
            x2 = x1 + self.__roi_size[0]  # get bottom right corner (x2,y2)
            y2 = y1 + self.__roi_size[1]
            print('({x1}, {y1})\t({x2}, {y2})'.format(x1=x1, y1=y1, x2=x2, y2=y2))
            return x1, y1, x2, y2

    def __wheel(self, event):
        """ Zoom with mouse wheel """
        #x = self.__canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        #y = self.__canvas.canvasy(event.y)
        #bbox = self.__canvas.coords(self.__container)  # get image area
        #if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]: pass  # Ok! Inside the image
        #else: return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down
            if int(self.__min_side * self.__imscale) < 30: return  # image is less than 30 pixels
            self.__imscale /= self.__delta
            scale          /= self.__delta
        if event.num == 4 or event.delta == 120:  # scroll up
            i = min(self.__canvas.winfo_width(), self.__canvas.winfo_height())
            if i < self.__imscale: return  # 1 pixel is bigger than the visible area
            self.__imscale *= self.__delta
            scale          *= self.__delta
        self.__canvas.scale('all', 0, 0, scale, scale)  # rescale all objects
        # Configure font size
        size = min(self.__font_max_size, max(self.__font_min_size,
                                             int(self.__text_size * self.__imscale)))
        self.__canvas.itemconfigure(self.__text_tag, font=self.__font.format(size=size))
        self.scale =  self.__imscale
        self.__show_image()  # zoom image and show it on the canvas

    def __set_focus(self, event):
        """ Set focus on the canvas and handle <Alt>+<Tab> switches between windows """
        self.__canvas.focus_set()
        #self.__motion(event)

    def __rid_focus(self, event=None):
        """ Hide region of interest and remove focus from the canvas """
        self.__canvas.itemconfigure(self.__text_warning, state='hidden')  # hide warning
        self.__canvas.itemconfigure(self.__roi_rect, state='hidden')  # hide roi
        self.__imframe.focus_set()  # remove focus from the canvas by setting it elsewhere

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if event.keycode in [68, 39, 102]:  # scroll right, keys 'd' or 'Right'
                self.__scroll_x('scroll',  1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:  # scroll left, keys 'a' or 'Left'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:  # scroll up, keys 'w' or 'Up'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:  # scroll down, keys 's' or 'Down'
                self.__scroll_y('scroll',  1, 'unit', event=event)

    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.__canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image
        if kwargs and kwargs['event']:
            self.__motion(kwargs['event'])

    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.__canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image
        if kwargs and kwargs['event']:
            self.__motion(kwargs['event'])
    def set_mode(self, mode):
        self.mode = mode
        if mode in ("brush_fg", "brush_bg"):
            self.display_mode = "overlay"
            self.__update_display_image()
            self.__show_image()
    def set_display_mode(self, mode):
        self.display_mode = mode
        self.__update_display_image()
        self.__show_image()

    def set_mask(self, mask):
        self.mask = mask.copy() if mask is not None else None
        self.__update_display_image()
        self.__show_image()

    def get_mask(self):
        return self.mask.copy() if self.mask is not None else None

    def clear_mask(self):
        if self.mask is not None:
            self.mask[:] = 0
        self.__update_display_image()
        self.__show_image()

    def set_brush_size(self, size):
        self.brush_size = max(1, int(size))

    def SetROI(self, roi):
        # Keep independent copies: editing curROICoord must never mutate history.
        self.measuredROI = copy.deepcopy(roi) if roi else []
        self.curROICoord = []
        # len(measuredROI) is the dedicated index for a new measurement.
        self.curIdx = len(self.measuredROI)
        self.__show_image()

    def DrawROI(self, idx, one, bbox, color):
        if len(one) > 0:
            ltx = one[0][0] * self.__imscale + bbox[0]
            lty = one[0][1] * self.__imscale + bbox[1]

            if idx == self.curIdx:
                c = self.__canvas.create_oval(
                    ltx - 3, lty - 3, ltx + 3, lty + 3,
                    width=1, outline='black', fill='green'
                )
                self.DrawList.append(c)

        if len(one) == 2:
            rbx = one[1][0] * self.__imscale + bbox[0]
            rby = one[1][1] * self.__imscale + bbox[1]

            if idx == self.curIdx:
                c = self.__canvas.create_oval(
                    rbx - 3, rby - 3, rbx + 3, rby + 3,
                    width=1, outline='black', fill='green'
                )
                self.DrawList.append(c)

            c = self.__canvas.create_line(ltx, lty, rbx, rby, fill=color)
            self.DrawList.append(c)

            distance = math.hypot(rbx - ltx, rby - lty) / self.__imscale
            x, y = max(ltx, rbx), max(lty, rby)

            if self.resolution > 0:
                t = self.__canvas.create_text(
                    x, y,
                    text=f"Stoma ID:  {idx + 1}",
                    anchor='w',
                    fill='blue',
                    font=('Helvetica', 10)
                )
                self.DrawList.append(t)

                t = self.__canvas.create_text(
                    x, y + 15,
                    text=f"Pixel:    {distance:.1f} pixel",
                    anchor='w',
                    fill='blue',
                    font=('Helvetica', 10)
                )
                self.DrawList.append(t)

                t = self.__canvas.create_text(
                    x, y + 30,
                    text=f"Distance: {distance * self.resolution:.1f} {self.unit}",
                    anchor='w',
                    fill='blue',
                    font=('Helvetica', 10)
                )
                self.DrawList.append(t)
            else:
                t = self.__canvas.create_text(
                    x, y,
                    text=f"Pixel: {distance:.1f} pixel",
                    fill='blue',
                    font=('Helvetica', 10)
                )
                self.DrawList.append(t)
                self.Pixel_Distance.set(f"{distance}")

    def ReFresh(self, event=None):
        self.__show_image(event)

    def __update_display_image(self):
        if self.__base_image is None:
            return

        base = self.__base_image.copy()

        if self.display_mode == "raw" or self.mask is None:
            self.__display_image = base
            self.__image = self.__display_image
            return

        if self.display_mode == "mask":
            mask_img = Image.fromarray(self.mask).convert("L")
            self.__display_image = mask_img.convert("RGB")
            self.__image = self.__display_image
            return

        base_np = np.array(base, dtype=np.uint8)
        mask_np = self.mask.astype(np.uint8)

        overlay = base_np.copy()
        overlay[mask_np > 0] = [255, 0, 0]

        alpha = self.overlay_alpha
        blended = base_np.copy()
        idx = mask_np > 0
        blended[idx] = (
            base_np[idx] * (1 - alpha) + overlay[idx] * alpha
        ).astype(np.uint8)

        self.__display_image = Image.fromarray(blended)
        self.__image = self.__display_image

    def __show_image(self, event=None):
        if self.__image is None:
            return

        for oneD in self.DrawList:
            self.__canvas.delete(oneD)
        self.DrawList.clear()

        bbox = self.__canvas.coords(self.__container)

        # Draw each ROI exactly once.  The selected/new current ROI is red;
        # stored historical ROIs are blue.  This prevents red/blue overdraw.
        current_is_visible = self.curIdx >= 0 and len(self.curROICoord) > 0
        if len(self.measuredROI) > 0:
            for idx, roi in enumerate(self.measuredROI):
                if current_is_visible and idx == self.curIdx:
                    continue
                self.DrawROI(idx, roi, bbox, 'blue')

        if current_is_visible:
            self.DrawROI(self.curIdx, self.curROICoord, bbox, 'red')

        box_image = self.__canvas.coords(self.__container)
        box_canvas = (
            self.__canvas.canvasx(0),
            self.__canvas.canvasy(0),
            self.__canvas.canvasx(self.__canvas.winfo_width()),
            self.__canvas.canvasy(self.__canvas.winfo_height())
        )

        box_img_int = tuple(map(int, box_image))

        box_scroll = [
            min(box_img_int[0], box_canvas[0]),
            min(box_img_int[1], box_canvas[1]),
            max(box_img_int[2], box_canvas[2]),
            max(box_img_int[3], box_canvas[3])
        ]

        if box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0] = box_img_int[0]
            box_scroll[2] = box_img_int[2]

        if box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1] = box_img_int[1]
            box_scroll[3] = box_img_int[3]

        self.__canvas.configure(scrollregion=tuple(map(int, box_scroll)))

        x1 = max(box_canvas[0] - box_image[0], 0)
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]

        if int(x2 - x1) > 0 and int(y2 - y1) > 0:
            image = self.__image.crop((
                int(x1 / self.__imscale),
                int(y1 / self.__imscale),
                int(x2 / self.__imscale),
                int(y2 / self.__imscale)
            ))

            imagetk = ImageTk.PhotoImage(
                image.resize((int(x2 - x1), int(y2 - y1)), self.__filter)
            )

            if self.__image_id is None:
                self.__image_id = self.__canvas.create_image(
                    max(box_canvas[0], box_img_int[0]),
                    max(box_canvas[1], box_img_int[1]),
                    anchor='nw',
                    image=imagetk
                )
            else:
                self.__canvas.coords(
                    self.__image_id,
                    max(box_canvas[0], box_img_int[0]),
                    max(box_canvas[1], box_img_int[1])
                )
                self.__canvas.itemconfig(self.__image_id, image=imagetk)

            self.__canvas.lower(self.__image_id)
            self.__canvas.imagetk = imagetk

    def destroy(self):
        if self.__image is not None and hasattr(self.__image, "close"):
            self.__image.close()
        self.__canvas.destroy()
        self.__imframe.destroy()



































































































































































