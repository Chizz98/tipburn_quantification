# GUI imports
import tkinter as tk
from tkinter import filedialog
import tkinter.ttk as ttk
from ctypes import windll
# Plotting imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
# Image handling
import utils
import segment
from skimage import io, color
import numpy as np


class MainWindow(tk.Tk):
    def __init__(self):
        windll.shcore.SetProcessDpiAwareness(1)
        super().__init__()
        self.title("LettuceSee")

        # Attributes
        self.filename = None
        self.im_arr = None
        self.mask_arr = None

        # Tk variables
        self.var_th1 = tk.IntVar()
        self.var_th2 = tk.IntVar()
        self.var_th3 = tk.IntVar()

        # Widgets
        self.fr_topbar = ttk.Frame( # Toplevel frame
            master=self
        )
        self.fr_topbar.grid(row=0, column=0, columnspan=2)
        self.bt_file = ttk.Button(  # Open file explorer
            master=self.fr_topbar,
            text="File explorer",
            command=self._bt_file
        )
        self.bt_file.grid(row=0, column=0)
        self.bt_hsv = ttk.Button(
            master=self.fr_topbar,
            text="Convert to HSV",
            state=tk.DISABLED,
            command=self._to_hsv
        )
        self.bt_hsv.grid(row=0, column=1)
        self.lb_watershed = ttk.Label(
            master=self.fr_topbar,
            text="Number of seeds:"
        )
        self.lb_watershed.grid(row=0, column=2)
        self.en_watershed = ttk.Entry(
            master=self.fr_topbar
        )
        self.en_watershed.grid(row=0, column=3)
        self.bt_watershed = ttk.Button(
            master=self.fr_topbar,
            text="Watershed blur",
            state=tk.DISABLED,
            command=self._watershed
        )
        self.bt_watershed.grid(row=0, column=4)
        for child in self.fr_topbar.winfo_children():
            child.grid_configure(padx=5, pady=3)

        self.fr_image = ttk.Frame(  # Frame for showing the images
            master=self,
            width="4i",
            height="4i"
        )
        self.fr_image.grid(row=1, column=0, sticky="ne")
        self.fr_mask = ttk.Frame(   # Frame for showing the mask
            master=self,
            width="4i",
            height="4i"
        )
        self.fr_mask.grid(row=1, column=1, sticky="nw")
        self.fr_threshold = ttk.Frame(  # Frame for threshold scales
            master=self,
            width="6i",
        )
        self.fr_threshold.grid(row=2, columnspan=2)
        self.lb_th1 = ttk.Label(
            master=self.fr_threshold,
            text="Channel 1 threshold",
        )
        self.lb_th1.grid(row=0, column=0)
        self.en_th1 = ttk.Entry(
            master=self.fr_threshold,
            textvariable=self.var_th1,
            state=tk.DISABLED
        )
        self.en_th1.grid(row=0, column=1)
        self.sc_th1 = tk.Scale(
            master=self.fr_threshold,
            variable=self.var_th1,
            length="7i",
            resolution=0.0001,
            orient=tk.HORIZONTAL,
            state=tk.DISABLED,
            takefocus=0,
            fg="lightgray"
        )
        self.sc_th1.grid(row=0, column=2)

        self.lb_th2 = ttk.Label(
            master=self.fr_threshold,
            text="Channel 2 threshold"
        )
        self.lb_th2.grid(row=1, column=0)
        self.en_th2 = ttk.Entry(
            master=self.fr_threshold,
            textvariable=self.var_th2,
            state=tk.DISABLED
        )
        self.en_th2.grid(row=1, column=1)
        self.sc_th2 = tk.Scale(
            master=self.fr_threshold,
            variable=self.var_th2,
            length="7i",
            resolution=0.0001,
            state=tk.DISABLED,
            orient=tk.HORIZONTAL,
            takefocus=0,
            fg="lightgray"
        )
        self.sc_th2.grid(row=1, column=2)
        self.lb_th3 = ttk.Label(
            master=self.fr_threshold,
            text="Channel 3 threshold"
        )
        self.lb_th3.grid(row=2, column=0)
        self.en_th3 = ttk.Entry(
            master=self.fr_threshold,
            textvariable=self.var_th3,
            state=tk.DISABLED
        )
        self.en_th3.grid(row=2, column=1)
        self.sc_th3 = tk.Scale(
            master=self.fr_threshold,
            variable=self.var_th3,
            length="7i",
            resolution=0.0001,
            orient=tk.HORIZONTAL,
            takefocus=0,
            fg="lightgray"
        )
        self.sc_th3.grid(row=2, column=2)
        for child in self.fr_threshold.winfo_children():
            child.grid_configure(padx=5, pady=2)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def _bt_file(self):
        """ Opens a file explorer """
        self.filename = filedialog.askopenfilename(
            title="Select image file",
            initialdir="/",
            filetypes=(
                ("PNG", "*.png*"),
                ("Fluorescence files", "*.fimg*")
            )
        )
        if self.filename.endswith(".fimg"):
            self.bt_watershed.configure(state=tk.DISABLED)
            self.bt_hsv.configure(state=tk.DISABLED)
            self.im_arr = utils.read_fimg(self.filename)
            self.mask_arr = np.zeros_like(self.im_arr)
        else:
            self.im_arr = io.imread(self.filename)
            if self.im_arr.shape[2] == 4:
                self.im_arr = color.rgba2rgb(self.im_arr)
            self.mask_arr = np.zeros(self.im_arr.shape[0:2])
            self.bt_watershed.configure(state="normal")
            self.bt_hsv.configure(state="normal")
        self._show_image(self.im_arr, self.fr_image)
        self._show_image(self.mask_arr, self.fr_mask, tb=False, mask=True)
        self._threshold_bars()

    def _show_image(self, im, master, tb=True, mask=False):
        """ Opens an image as np array """
        for child in master.winfo_children():
            child.destroy()
        fig = plt.Figure(
            figsize=(4, 4),
            dpi=110,
            tight_layout=True
        )
        image = fig.add_subplot()
        if mask:
            image.imshow(im, cmap="cividis", vmin=0, vmax=1)
        else:
            image.imshow(im)
        image.axis("off")
        canvas = FigureCanvasTkAgg(figure=fig, master=master)
        canvas.draw()
        # Toolbar, gets packed with canvas
        if tb:
            toolbar = NavigationToolbar2Tk(canvas, window=master)
        canvas_w = canvas.get_tk_widget()
        canvas_w.pack()

    def _threshold_bars(self):
        """ Opens a thresholding bar for each channel in the image """
        if self.im_arr.ndim == 2:
            self.sc_th1.configure(
                state="normal",
                from_=0,
                to=self.im_arr.max(),
                fg="black"
            )
            self.sc_th1.bind("<ButtonRelease-1>", self._update_mask)
            self.en_th1.bind("<Return>", self._update_mask)
            self.sc_th2.configure(state=tk.DISABLED)
            self.sc_th2.unbind("<ButtonRelease-1>")
            self.en_th2.configure(state=tk.DISABLED)
            self.var_th2.set(0)
            self.sc_th3.configure(state=tk.DISABLED)
            self.en_th3.configure(state=tk.DISABLED)
            self.sc_th3.unbind("<ButtonRelease-1>")
            self.var_th3.set(0)
        else:
            self.var_th1.set(0)
            self.sc_th1.configure(
                state="normal",
                from_=0,
                to=self.im_arr[:, :, 0].max(),
                fg="black"
            )
            self.sc_th1.bind("<ButtonRelease-1>", self._update_mask)
            self.en_th1.configure(
                state="normal"
            )
            self.en_th1.bind("<Return>", self._update_mask)
            self.var_th2.set(0)
            self.sc_th2.configure(
                state="normal",
                from_=0,
                to=self.im_arr[:, :, 1].max(),
                fg="black"
            )
            self.sc_th2.bind("<ButtonRelease-1>", self._update_mask)
            self.en_th2.configure(
                state="normal"
            )
            self.en_th2.bind("<Return>", self._update_mask)
            self.var_th3.set(0)
            self.sc_th3.configure(
                state="normal",
                from_=0,
                to=self.im_arr[:, :, 2].max(),
                fg="black"
            )
            self.sc_th3.bind("<ButtonRelease-1>", self._update_mask)
            self.en_th3.configure(
                state="normal"
            )
            self.en_th3.bind("<Return>", self._update_mask)

    def _update_mask(self, *args):
        if self.im_arr.ndim == 2:
            self.mask_arr = self.im_arr >= float(self.sc_th1.get())
        else:
            th1 = float(self.sc_th1.get())
            th2 = float(self.sc_th2.get())
            th3 = float(self.sc_th3.get())
            self.mask_arr = segment.multichannel_threshold(
                self.im_arr, th1, th2, th3
            )
        self._show_image(self.mask_arr, self.fr_mask, tb=False, mask=True)

    def _to_hsv(self):
        self.im_arr = color.rgb2hsv(self.im_arr)
        self._show_image(self.im_arr, self.fr_image)
        self._threshold_bars()
        self.bt_hsv.configure(state=tk.DISABLED)
        self.mask_arr = np.ones(self.im_arr.shape[0:2])
        self._show_image(self.mask_arr, self.fr_mask, tb=False)

    def _watershed(self):
        seeds = int(self.en_watershed.get())
        self.im_arr = segment.watershed_blur(self.im_arr, seeds)
        self._show_image(self.im_arr, self.fr_image)


def main():
    window = MainWindow()
    #window.state("zoomed")
    window.iconbitmap(r"C:\Users\chris\Documents\GitHub\tipburn_quantification"
                      r"\logo\lettuce.ico")
    window.mainloop()


if __name__ == "__main__":
    main()

