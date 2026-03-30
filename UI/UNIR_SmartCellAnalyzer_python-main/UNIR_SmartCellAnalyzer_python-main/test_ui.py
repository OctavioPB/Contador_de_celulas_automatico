import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Cell Analyzer - Test")

ttk.Label(root, text="Tkinter funcionando correctamente").pack(padx=20, pady=20)
ttk.Button(root, text="Cerrar", command=root.destroy).pack(pady=10)

root.mainloop()
