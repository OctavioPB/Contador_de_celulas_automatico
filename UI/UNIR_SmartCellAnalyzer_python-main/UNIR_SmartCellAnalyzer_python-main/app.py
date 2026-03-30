import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import cv2
import numpy as np
from PIL import Image, ImageTk
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as rl_canvas

# ---------------------------------------------------------------------------
# sys.path: subir 3 niveles desde app.py para llegar a la raiz del proyecto
# ---------------------------------------------------------------------------
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_APP_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_DETECTION_MODEL = os.path.join(_PROJECT_ROOT, "Models", "model_base_3A")
_ORIENTATION_MODEL = os.path.join(
    _PROJECT_ROOT, "Orientation", "Orientador_De_Fibras_CNN",
    "models", "cnn_fiber_orientation.pth"
)

APP_TITLE = "Smart Cell AI Analysis Studio (Beta)"

# Definicion de los 7 tabs del visor de imagen
_IMG_TABS = [
    ("original",  "1 Original"),
    ("prep",      "2 Prep."),
    ("seg",       "3 Segm."),
    ("cells",     "4 Celulas"),
    ("areas",     "5 Areas"),
    ("orient",    "6 Orient."),
    ("all",       "Todas"),
]


class App(tb.Window):
    def __init__(self):
        super().__init__(themename="flatly")
        self.title(APP_TITLE)
        self.geometry("1280x720")
        self.minsize(1024, 600)

        # Estado imagen
        self.current_image = None
        self.current_image_tk = None
        self.current_image_path = None

        # Estado zoom / pan
        self._zoom_level = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._original_image = None
        self._drag_start = None
        self._zoom_label = None

        # Estado analisis
        self._last_result = None
        self._active_img_tab = "all"
        self._img_tab_btns = {}
        self._analysis_cache = ""
        self._summary_vars = {}
        self._tv_orient = None
        self._tv_detail = None
        self._tv_features = None
        self._analyze_btn = None

        self.root = tb.Frame(self, padding=10)
        self.root.pack(fill=BOTH, expand=True)

        self._create_menu()
        self._create_body()

    # =========================================================================
    # Menu
    # =========================================================================
    def _create_menu(self):
        menubar = tk.Menu(self)
        menu_archivo = tk.Menu(menubar, tearoff=0)
        menu_archivo.add_command(label="Abrir imagen...", command=self.open_image)
        menu_archivo.add_separator()
        menu_archivo.add_command(label="Salir", command=self.destroy)
        menu_exportar = tk.Menu(menubar, tearoff=0)
        menu_exportar.add_command(label="Exportar a PDF...", command=self.export_pdf)
        menubar.add_cascade(label="Archivo", menu=menu_archivo)
        menubar.add_cascade(label="Exportar", menu=menu_exportar)
        self.config(menu=menubar)

    # =========================================================================
    # Layout principal
    # =========================================================================
    def _create_body(self):
        body = tb.Frame(self.root)
        body.pack(fill=BOTH, expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        self._build_left_panel(body)
        self._build_right_panel(body)

    # -------------------------------------------------------------------------
    # Panel izquierdo: tabs de imagen + canvas + barra de zoom
    # -------------------------------------------------------------------------
    def _build_left_panel(self, parent):
        left_card = tb.Frame(parent, bootstyle="light", padding=10)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        # --- Tabs del visor de imagen (7 botones) ---
        tabs_bar = tb.Frame(left_card)
        tabs_bar.pack(fill=X, pady=(0, 6))

        for key, lbl in _IMG_TABS:
            style = "primary" if key == "all" else "secondary-outline"
            btn = tb.Button(
                tabs_bar, text=lbl, bootstyle=style, padding=(5, 2),
                command=lambda k=key: self._switch_image_tab(k),
            )
            btn.pack(side=LEFT, padx=1)
            self._img_tab_btns[key] = btn

        # --- Canvas ---
        left_inner = tb.Frame(left_card, bootstyle="secondary", padding=2)
        left_inner.pack(fill=BOTH, expand=True)

        self.canvas = tk.Canvas(left_inner, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # --- Barra de zoom ---
        zoom_bar = tb.Frame(left_card)
        zoom_bar.pack(fill=X, pady=(8, 0))
        tb.Label(zoom_bar, text="Zoom:", bootstyle="secondary").pack(side=LEFT)
        tb.Button(zoom_bar, text="+",   width=3, bootstyle="secondary", command=self.zoom_in).pack(side=LEFT, padx=(8, 4))
        tb.Button(zoom_bar, text="-",   width=3, bootstyle="secondary", command=self.zoom_out).pack(side=LEFT, padx=(0, 4))
        tb.Button(zoom_bar, text="[ ]", width=4, bootstyle="secondary", command=self.zoom_reset).pack(side=LEFT)
        self._zoom_label = tb.Label(zoom_bar, text="Ajuste", bootstyle="secondary", width=7)
        self._zoom_label.pack(side=LEFT, padx=(10, 0))

        self._draw_mock_view()

    # -------------------------------------------------------------------------
    # Panel derecho: header + Notebook con 3 tabs + botones
    # -------------------------------------------------------------------------
    def _build_right_panel(self, parent):
        right = tb.Frame(parent)
        right.grid(row=0, column=1, sticky="nsew")

        analysis_card = tb.Frame(right, bootstyle="light", padding=10)
        analysis_card.pack(fill=BOTH, expand=True)

        # Header
        header = tb.Frame(analysis_card, bootstyle="primary", padding=(10, 8))
        header.pack(fill=X)
        tb.Label(
            header, text="Analisis", bootstyle="inverse-primary",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor=W)

        content = tb.Frame(analysis_card, padding=(6, 8))
        content.pack(fill=BOTH, expand=True)

        # --- Notebook de resultados (3 tabs) ---
        self._results_nb = tb.Notebook(content, bootstyle="primary")
        self._results_nb.pack(fill=BOTH, expand=True, pady=(0, 10))

        # Tab 1: Resumen
        tab_summary = tb.Frame(self._results_nb, padding=12)
        self._results_nb.add(tab_summary, text="  Resumen  ")
        self._build_summary_tab(tab_summary)

        # Tab 2: Orientacion
        tab_orient = tb.Frame(self._results_nb, padding=4)
        self._results_nb.add(tab_orient, text=" Orientacion ")
        self._tv_orient = self._make_scrollable_treeview(
            tab_orient,
            columns=("#", "CNN (deg)", "Feature (deg)", "Metodo"),
            widths=(45, 90, 105, 75),
        )

        # Tab 3: Detalle
        tab_detail = tb.Frame(self._results_nb, padding=4)
        self._results_nb.add(tab_detail, text="  Detalle  ")
        self._tv_detail = self._make_scrollable_treeview(
            tab_detail,
            columns=("#", "Area (px)", "CNN (deg)", "Feat. (deg)", "Metodo"),
            widths=(40, 85, 85, 95, 70),
        )

        # Tab 4: Features
        _FEAT_COLS = ("label", "area", "perimeter", "eccentricity", "solidity",
                      "major_axis", "minor_axis", "orientation", "centroid_x", "centroid_y")
        _FEAT_WIDTHS = (45, 60, 75, 90, 70, 80, 80, 85, 80, 80)
        tab_features = tb.Frame(self._results_nb, padding=4)
        self._results_nb.add(tab_features, text=" Features ")
        self._tv_features = self._make_bidir_treeview(tab_features, _FEAT_COLS, _FEAT_WIDTHS)

        # Botones de accion
        self._analyze_btn = tb.Button(
            content, text="Analizar", bootstyle="primary",
            command=self.analyze,
        )
        self._analyze_btn.pack(fill=X, ipady=6, pady=(0, 6))
        tb.Button(content, text="Guardar Analisis", bootstyle="secondary",
                  command=self.save_analysis).pack(fill=X, ipady=4, pady=(0, 6))
        tb.Button(content, text="Exportar a PDF", bootstyle="secondary",
                  command=self.export_pdf).pack(fill=X, ipady=4)

    def _build_summary_tab(self, parent):
        self._summary_vars = {
            "total_cells": tk.StringVar(value="—"),
            "mean_area":   tk.StringVar(value="—"),
            "std_area":    tk.StringVar(value="—"),
        }
        rows = [
            ("Total celulas",          "total_cells"),
            ("Area media (px)",        "mean_area"),
            ("Desv. estandar (px)",    "std_area"),
        ]
        for i, (label, key) in enumerate(rows):
            tb.Label(parent, text=label,
                     font=("Segoe UI", 10, "bold")).grid(row=i, column=0, sticky=W, padx=(0, 20), pady=6)
            tb.Label(parent, textvariable=self._summary_vars[key],
                     font=("Segoe UI", 12)).grid(row=i, column=1, sticky=W, pady=6)

        # Mensaje inicial
        self._summary_hint = tb.Label(
            parent, text="Abre una imagen y pulsa Analizar.",
            bootstyle="secondary", font=("Segoe UI", 9, "italic"),
        )
        self._summary_hint.grid(row=len(rows), column=0, columnspan=2, sticky=W, pady=(16, 0))

    def _make_bidir_treeview(self, parent, columns, widths):
        """Treeview con scrollbars vertical y horizontal."""
        frame = tb.Frame(parent)
        frame.pack(fill=BOTH, expand=True)

        vsb = tb.Scrollbar(frame, orient="vertical", bootstyle="secondary-round")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb = tb.Scrollbar(frame, orient="horizontal", bootstyle="secondary-round")
        hsb.grid(row=1, column=0, sticky="ew")

        tv = tb.Treeview(
            frame, columns=columns, show="headings",
            yscrollcommand=vsb.set, xscrollcommand=hsb.set,
            bootstyle="primary",
        )
        tv.grid(row=0, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        vsb.configure(command=tv.yview)
        hsb.configure(command=tv.xview)

        for col, w in zip(columns, widths):
            tv.heading(col, text=col)
            tv.column(col, width=w, anchor=CENTER, stretch=False, minwidth=w)

        return tv

    def _make_scrollable_treeview(self, parent, columns, widths):
        frame = tb.Frame(parent)
        frame.pack(fill=BOTH, expand=True)

        vsb = tb.Scrollbar(frame, orient="vertical", bootstyle="secondary-round")
        vsb.pack(side=RIGHT, fill=Y)

        tv = tb.Treeview(
            frame, columns=columns, show="headings",
            yscrollcommand=vsb.set, bootstyle="primary",
        )
        tv.pack(fill=BOTH, expand=True)
        vsb.configure(command=tv.yview)

        for col, w in zip(columns, widths):
            tv.heading(col, text=col)
            tv.column(col, width=w, anchor=CENTER, stretch=True)

        return tv

    # =========================================================================
    # Tabs del visor de imagen
    # =========================================================================
    def _switch_image_tab(self, key: str):
        self._active_img_tab = key
        for k, btn in self._img_tab_btns.items():
            btn.configure(bootstyle="primary" if k == key else "secondary-outline")

        if self._last_result is None:
            return

        img_map = {
            "original": self._last_result.img_original,
            "prep":     self._last_result.img_preprocessed,
            "seg":      self._last_result.img_segmentation,
            "cells":    self._last_result.img_cells,
            "areas":    self._last_result.img_area_hist,
            "orient":   self._last_result.img_orient_hist,
            "all":      self._last_result.report_figure,
        }
        img = img_map.get(key)
        if img is not None:
            self.display_image(img)

    # =========================================================================
    # Mock view (canvas vacio)
    # =========================================================================
    def _draw_mock_view(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 760
        h = self.canvas.winfo_height() or 480
        self.canvas.create_rectangle(
            20, 20, w - 20, h - 20,
            fill="#5b6f86", outline="#3a3a3a", width=2,
        )

    # =========================================================================
    # Abrir imagen
    # =========================================================================
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Abrir imagen",
            filetypes=[
                ("Imagenes", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not path:
            return

        self.current_image_path = path
        self.current_image = cv2.imread(path)

        if self.current_image is None:
            messagebox.showerror("Error", "No se pudo cargar la imagen.")
            return

        # Resetear estado de analisis anterior
        self._last_result = None
        self._analysis_cache = ""
        for k, btn in self._img_tab_btns.items():
            btn.configure(bootstyle="secondary-outline")

        self.display_image(self.current_image)

    # =========================================================================
    # Visor de imagen con zoom / pan
    # =========================================================================
    def display_image(self, img_cv):
        """Carga imagen BGR, resetea zoom y renderiza."""
        self._original_image = img_cv.copy()
        self._zoom_level = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._render_image()

    def _render_image(self):
        if self._original_image is None:
            return

        img = self._original_image
        H, W = img.shape[:2]
        cw = self.canvas.winfo_width() or 760
        ch = self.canvas.winfo_height() or 480

        fit_scale = min(cw / W, ch / H)
        scale = fit_scale * self._zoom_level

        new_w = max(1, int(W * scale))
        new_h = max(1, int(H * scale))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((new_w, new_h), Image.LANCZOS)

        canvas_img = Image.new("RGB", (cw, ch), (43, 43, 43))

        if new_w <= cw and new_h <= ch:
            self._pan_x = 0
            self._pan_y = 0
            canvas_img.paste(img_pil, ((cw - new_w) // 2, (ch - new_h) // 2))
        else:
            cx = new_w // 2 + self._pan_x
            cy = new_h // 2 + self._pan_y
            cx = max(cw // 2, min(new_w - cw // 2, cx))
            cy = max(ch // 2, min(new_h - ch // 2, cy))
            self._pan_x = cx - new_w // 2
            self._pan_y = cy - new_h // 2
            x1 = cx - cw // 2
            y1 = cy - ch // 2
            canvas_img.paste(img_pil.crop((x1, y1, x1 + cw, y1 + ch)), (0, 0))

        self.current_image_tk = ImageTk.PhotoImage(canvas_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.current_image_tk)
        self._update_zoom_label()

    def _update_zoom_label(self):
        if self._zoom_label is None or self._original_image is None:
            return
        H, W = self._original_image.shape[:2]
        cw = self.canvas.winfo_width() or 760
        ch = self.canvas.winfo_height() or 480
        pct = int(min(cw / W, ch / H) * self._zoom_level * 100)
        self._zoom_label.configure(text=f"{pct} %")

    # =========================================================================
    # Analisis
    # =========================================================================
    def analyze(self):
        if self.current_image is None or self.current_image_path is None:
            messagebox.showwarning("Sin imagen", "Primero abre una imagen.")
            return

        import glob
        if not glob.glob(_DETECTION_MODEL + "*"):
            messagebox.showerror(
                "Modelo no encontrado",
                f"No se encontro el modelo Cellpose en:\n{_DETECTION_MODEL}\n\n"
                "Descargalo y coloca en la carpeta Models/.",
            )
            return

        self._set_status("Analizando... (puede tardar varios segundos)")
        if self._analyze_btn:
            self._analyze_btn.configure(state="disabled")

        def worker():
            try:
                from Core.pipeline import run_analysis
                result = run_analysis(
                    self.current_image_path,
                    _DETECTION_MODEL,
                    _ORIENTATION_MODEL,
                )
                self.after(0, lambda: self._on_analysis_complete(result))
            except Exception as e:
                import traceback
                self.after(0, lambda: self._on_analysis_error(traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _on_analysis_complete(self, result):
        self._last_result = result

        # Mostrar vista "Todas" por defecto
        self._switch_image_tab("all")

        # Poblar tab Resumen
        self._summary_vars["total_cells"].set(str(result.count))
        self._summary_vars["mean_area"].set(f"{result.mean_area:.1f}")
        self._summary_vars["std_area"].set(f"{result.std_area:.1f}")
        if hasattr(self, "_summary_hint"):
            self._summary_hint.configure(text="")

        # Poblar tab Orientacion
        for row in self._tv_orient.get_children():
            self._tv_orient.delete(row)
        for i, (cnn_a, feat_a, fb) in enumerate(
            zip(result.angles, result.feature_angles, result.used_fallback), 1
        ):
            self._tv_orient.insert(
                "", "end",
                values=(i, f"{cnn_a:.1f}", f"{feat_a:.1f}", "Elipse" if fb else "CNN"),
            )

        # Poblar tab Detalle
        for row in self._tv_detail.get_children():
            self._tv_detail.delete(row)
        for i, (area, cnn_a, feat_a, fb) in enumerate(
            zip(result.areas, result.angles, result.feature_angles, result.used_fallback), 1
        ):
            self._tv_detail.insert(
                "", "end",
                values=(i, f"{area:.0f}", f"{cnn_a:.1f}", f"{feat_a:.1f}", "Elipse" if fb else "CNN"),
            )

        # Poblar tab Features
        for row in self._tv_features.get_children():
            self._tv_features.delete(row)
        for feat in result.cell_features:
            self._tv_features.insert(
                "", "end",
                values=(
                    feat["label"],
                    feat["area"],
                    f"{feat['perimeter']:.3f}",
                    f"{feat['eccentricity']:.4f}",
                    f"{feat['solidity']:.4f}",
                    f"{feat['major_axis']:.4f}",
                    f"{feat['minor_axis']:.4f}",
                    f"{feat['orientation']:.4f}",
                    f"{feat['centroid_x']:.4f}",
                    f"{feat['centroid_y']:.4f}",
                ),
            )

        # Cache de texto para guardar/exportar
        self._analysis_cache = self._build_text_report(result)

        self._set_status(f"Listo  —  {result.count} celulas detectadas")
        if self._analyze_btn:
            self._analyze_btn.configure(state="normal")

    def _on_analysis_error(self, message: str):
        messagebox.showerror("Error en el analisis", message)
        self._set_status("Error durante el analisis")
        if self._analyze_btn:
            self._analyze_btn.configure(state="normal")

    def _set_status(self, text: str):
        self.title(f"{APP_TITLE}  —  {text}")

    def _build_text_report(self, result) -> str:
        lines = [
            "=== RESUMEN ===",
            f"total_cells          {result.count}",
            f"mean_area            {result.mean_area:.1f} px",
            f"std_area             {result.std_area:.1f} px",
            "",
            "=== ORIENTACION ===",
            f"{'#':<5} {'CNN (deg)':<14} {'Feature (deg)':<14} Metodo",
            "-" * 48,
        ]
        for i, (a, f, fb) in enumerate(
            zip(result.angles, result.feature_angles, result.used_fallback), 1
        ):
            lines.append(f"{i:<5} {a:<14.1f} {f:<14.1f} {'Elipse' if fb else 'CNN'}")
        lines += [
            "",
            "=== DETALLE ===",
            f"{'#':<5} {'Area (px)':<12} {'CNN (deg)':<13} {'Feat (deg)':<13} Metodo",
            "-" * 56,
        ]
        for i, (area, a, f, fb) in enumerate(
            zip(result.areas, result.angles, result.feature_angles, result.used_fallback), 1
        ):
            lines.append(f"{i:<5} {area:<12.0f} {a:<13.1f} {f:<13.1f} {'Elipse' if fb else 'CNN'}")
        return "\n".join(lines)

    # =========================================================================
    # Guardar y exportar
    # =========================================================================
    def save_analysis(self):
        content = self._analysis_cache.strip()
        if not content:
            messagebox.showwarning("Sin contenido", "No hay analisis para guardar.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar analisis",
            defaultextension=".txt",
            filetypes=[("Texto", "*.txt"), ("CSV", "*.csv"), ("Todos", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Guardado", f"Guardado en:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_pdf(self):
        content = self._analysis_cache.strip()
        if not content:
            messagebox.showwarning("Sin contenido", "No hay analisis para exportar.")
            return
        path = filedialog.asksaveasfilename(
            title="Exportar a PDF",
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("Todos", "*.*")],
        )
        if not path:
            return
        try:
            c = rl_canvas.Canvas(path, pagesize=letter)
            width, height = letter
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, height - 50, "Analisis de Imagen")
            c.setFont("Helvetica", 9)
            c.drawString(50, height - 66, f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            text_obj = c.beginText(50, height - 90)
            text_obj.setFont("Courier", 8)
            for line in content.splitlines():
                text_obj.textLine(line)
            c.drawText(text_obj)
            c.showPage()
            c.save()
            messagebox.showinfo("Exportado", f"PDF generado en:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # =========================================================================
    # Zoom y pan
    # =========================================================================
    _ZOOM_STEP = 1.25
    _ZOOM_MAX = 8.0

    def zoom_in(self):
        if self._original_image is None:
            return
        self._zoom_level = min(self._zoom_level * self._ZOOM_STEP, self._ZOOM_MAX)
        self._render_image()

    def zoom_out(self):
        if self._original_image is None:
            return
        self._zoom_level = max(self._zoom_level / self._ZOOM_STEP, 1.0)
        if self._zoom_level == 1.0:
            self._pan_x = self._pan_y = 0
        self._render_image()

    def zoom_reset(self):
        if self._original_image is None:
            return
        self._zoom_level = 1.0
        self._pan_x = self._pan_y = 0
        self._render_image()

    def _on_mousewheel(self, event):
        if self._original_image is None:
            return
        if event.delta > 0:
            self._zoom_level = min(self._zoom_level * self._ZOOM_STEP, self._ZOOM_MAX)
        else:
            self._zoom_level = max(self._zoom_level / self._ZOOM_STEP, 1.0)
            if self._zoom_level == 1.0:
                self._pan_x = self._pan_y = 0
        self._render_image()

    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y)

    def _on_drag_move(self, event):
        if self._drag_start is None or self._original_image is None or self._zoom_level <= 1.0:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        self._pan_x -= dx
        self._pan_y -= dy
        self._render_image()

    def _on_canvas_resize(self, event):
        if self._original_image is not None:
            self._render_image()
        else:
            self._draw_mock_view()


if __name__ == "__main__":
    app = App()
    app.mainloop()
