"""
Reusable directory/file browser widget for customtkinter.
"""

import customtkinter as ctk
from tkinter import filedialog


class DirBrowser(ctk.CTkFrame):
    """A labeled entry + Browse button that returns a directory path."""

    def __init__(self, master, label_text="Directory:", button_text="Browse",
                 mode="directory", filetypes=None, **kwargs):
        super().__init__(master, **kwargs)

        self.mode = mode
        self.filetypes = filetypes or [("All files", "*.*")]
        self.path_var = ctk.StringVar(value="")

        self.label = ctk.CTkLabel(self, text=label_text)
        self.label.pack(side="left", padx=(0, 5))

        self.entry = ctk.CTkEntry(self, textvariable=self.path_var, width=400)
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.button = ctk.CTkButton(self, text=button_text, width=80,
                                    command=self._browse)
        self.button.pack(side="left")

    def _browse(self):
        if self.mode == "directory":
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename(filetypes=self.filetypes)

        if path:
            self.path_var.set(path)

    def get(self):
        return self.path_var.get()

    def set(self, value):
        self.path_var.set(value)
