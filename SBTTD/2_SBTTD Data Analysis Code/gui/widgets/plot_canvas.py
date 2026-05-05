"""
Reusable Matplotlib canvas widget for embedding plots in customtkinter frames.
"""

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class PlotCanvas(ctk.CTkFrame):
    """Embeds a matplotlib Figure inside a CTkFrame with a navigation toolbar."""

    def __init__(self, master, figsize=(8, 5), dpi=100, **kwargs):
        super().__init__(master, **kwargs)

        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

    def clear(self):
        self.ax.cla()
        self.canvas.draw()

    def draw(self):
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    def get_fig_ax(self):
        return self.fig, self.ax

    def replace_figure(self, fig):
        """Replace internal figure with an externally-created one."""
        self.canvas.get_tk_widget().destroy()
        self.toolbar.destroy()

        self.fig = fig
        self.ax = fig.axes[0] if fig.axes else None

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
