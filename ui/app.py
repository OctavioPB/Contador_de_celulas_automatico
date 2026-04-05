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
# sys.path: subir 1 nivel desde ui/app.py para llegar a la raiz del proyecto
# ---------------------------------------------------------------------------
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_DETECTION_MODEL = os.path.join(_PROJECT_ROOT, "models", "model_base_3B")
_ORIENTATION_MODEL = os.path.join(
    _PROJECT_ROOT, "orientation", "models", "cnn_fiber_orientation.pth"
)

APP_TITLE = "Smart Cell AI Analysis Studio (Beta)"

# Definicion de los 8 tabs del visor de imagen
_IMG_TABS = [
    ("original",  "1 Original"),
    ("prep",      "2 Prep."),
    ("seg",       "3 Segm."),
    ("cells",     "4 Celulas"),
    ("boundary",  "5 Limites"),
    ("areas",     "6 Areas"),
    ("orient",    "7 Orient."),
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
        self._angles_csv_btn = None

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
        tb.Button(
            tab_orient, text="Descargar CSV", bootstyle="success-outline",
            command=lambda: self._export_treeview_csv(self._tv_orient, "orientacion.csv"),
        ).pack(fill=X, pady=(4, 0))

        # Tab 3: Detalle
        tab_detail = tb.Frame(self._results_nb, padding=4)
        self._results_nb.add(tab_detail, text="  Detalle  ")
        self._tv_detail = self._make_scrollable_treeview(
            tab_detail,
            columns=("#", "Area (px)", "CNN (deg)", "Feat. (deg)", "Metodo"),
            widths=(40, 85, 85, 95, 70),
        )
        tb.Button(
            tab_detail, text="Descargar CSV", bootstyle="success-outline",
            command=lambda: self._export_treeview_csv(self._tv_detail, "detalle.csv"),
        ).pack(fill=X, pady=(4, 0))

        # Tab 4: Features
        _FEAT_COLS = ("label", "area", "perimeter", "eccentricity", "solidity",
                      "major_axis", "minor_axis", "orientation",
                      "circularity", "feret_diameter",
                      "centroid_x", "centroid_y")
        _FEAT_WIDTHS = (45, 60, 75, 90, 70, 80, 80, 85,
                        80, 90,
                        80, 80)
        tab_features = tb.Frame(self._results_nb, padding=4)
        self._results_nb.add(tab_features, text=" Features ")
        self._tv_features = self._make_bidir_treeview(tab_features, _FEAT_COLS, _FEAT_WIDTHS)
        tb.Button(
            tab_features, text="Descargar CSV", bootstyle="success-outline",
            command=lambda: self._export_treeview_csv(self._tv_features, "cell_features.csv"),
        ).pack(fill=X, pady=(4, 0))

        # Botones de accion
        self._analyze_btn = tb.Button(
            content, text="Analizar", bootstyle="primary",
            command=self.analyze,
        )
        self._analyze_btn.pack(fill=X, ipady=6, pady=(0, 6))
        self._progress_bar = tb.Progressbar(
            content, mode="indeterminate", bootstyle="primary"
        )
        self._progress_bar.pack(fill=X, pady=(0, 6))
        self._angles_csv_btn = tb.Button(
            content, text="Descargar CSV de Angulos", bootstyle="success",
            command=self.download_angles_csv, state="disabled",
        )
        self._angles_csv_btn.pack(fill=X, ipady=4, pady=(0, 6))
        tb.Button(content, text="Guardar Analisis", bootstyle="secondary",
                  command=self.save_analysis).pack(fill=X, ipady=4, pady=(0, 6))
        tb.Button(content, text="Guardar Imagen Procesada", bootstyle="secondary",
                  command=self.save_processed_image).pack(fill=X, ipady=4, pady=(0, 6))
        tb.Button(content, text="Exportar a PDF", bootstyle="secondary",
                  command=self.export_pdf).pack(fill=X, ipady=4)

    def _build_summary_tab(self, parent):
        self._summary_vars = {
            "total_cells": tk.StringVar(value="—"),
            "mean_area":   tk.StringVar(value="—"),
            "std_area":    tk.StringVar(value="—"),
            "snr_before":  tk.StringVar(value="—"),
            "snr_after":   tk.StringVar(value="—"),
            "cv_after":    tk.StringVar(value="—"),
            "p25_area":    tk.StringVar(value="—"),
            "p50_area":    tk.StringVar(value="—"),
            "p75_area":    tk.StringVar(value="—"),
        }
        rows = [
            ("Total celulas",          "total_cells"),
            ("Area media (px)",        "mean_area"),
            ("Desv. estandar (px)",    "std_area"),
            ("SNR antes preprocesar",   "snr_before"),
            ("SNR despues preprocesar", "snr_after"),
            ("CV intensidad (post)",    "cv_after"),
            ("P25 area (px)",           "p25_area"),
            ("Mediana area (px)",       "p50_area"),
            ("P75 area (px)",           "p75_area"),
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
            "boundary": self._last_result.img_boundary,
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
                "Descargalo y coloca en la carpeta models/.",
            )
            return

        self._set_status("Analizando... (puede tardar varios segundos)")
        if self._analyze_btn:
            self._analyze_btn.configure(state="disabled")
        self._progress_bar.start(10)

        def worker():
            try:
                from core.pipeline import run_analysis
                result = run_analysis(
                    self.current_image_path,
                    _DETECTION_MODEL,
                    _ORIENTATION_MODEL,
                )
                self.after(0, lambda: self._on_analysis_complete(result))
            except Exception:
                import traceback
                tb = traceback.format_exc()
                self.after(0, lambda: self._on_analysis_error(tb))

        threading.Thread(target=worker, daemon=True).start()

    def _on_analysis_complete(self, result):
        self._last_result = result

        # Mostrar vista "Todas" por defecto
        self._switch_image_tab("all")

        # Poblar tab Resumen
        self._summary_vars["total_cells"].set(str(result.count))
        self._summary_vars["mean_area"].set(f"{result.mean_area:.1f}")
        self._summary_vars["std_area"].set(f"{result.std_area:.1f}")
        pm = result.preprocessing_metrics
        self._summary_vars["snr_before"].set(f"{pm.get('snr_before', 0):.2f}")
        self._summary_vars["snr_after"].set(f"{pm.get('snr_after', 0):.2f}")
        self._summary_vars["cv_after"].set(f"{pm.get('cv_after', 0):.4f}")
        self._summary_vars["p25_area"].set(f"{result.p25_area:.1f}")
        self._summary_vars["p50_area"].set(f"{result.p50_area:.1f}")
        self._summary_vars["p75_area"].set(f"{result.p75_area:.1f}")
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
                    f"{feat['circularity']:.4f}",
                    f"{feat['feret_diameter']:.4f}",
                    f"{feat['centroid_x']:.4f}",
                    f"{feat['centroid_y']:.4f}",
                ),
            )

        # Cache de texto para guardar/exportar
        self._analysis_cache = self._build_text_report(result)

        # Activar botón de descarga de CSV de ángulos si el archivo existe
        if self._angles_csv_btn:
            has_csv = bool(result.angles_csv_path and os.path.isfile(result.angles_csv_path))
            self._angles_csv_btn.configure(state="normal" if has_csv else "disabled")

        self._progress_bar.stop()
        self._set_status(f"Listo  —  {result.count} celulas detectadas")
        if self._analyze_btn:
            self._analyze_btn.configure(state="normal")

    def _on_analysis_error(self, message: str):
        self._progress_bar.stop()
        messagebox.showerror("Error en el analisis", message)
        self._set_status("Error durante el analisis")
        if self._analyze_btn:
            self._analyze_btn.configure(state="normal")

    def save_processed_image(self):
        if self._last_result is None:
            messagebox.showwarning("Sin datos", "Ejecuta el analisis primero.")
            return
        img_map = {
            "original":  self._last_result.img_original,
            "prep":      self._last_result.img_preprocessed,
            "seg":       self._last_result.img_segmentation,
            "cells":     self._last_result.img_cells,
            "boundary":  self._last_result.img_boundary,
            "areas":     self._last_result.img_area_hist,
            "orient":    self._last_result.img_orient_hist,
            "all":       self._last_result.report_figure,
        }
        img = img_map.get(self._active_img_tab)
        if img is None:
            messagebox.showwarning("Sin imagen", "No hay imagen en el tab activo.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar imagen procesada",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif"), ("Todos", "*.*")],
        )
        if not path:
            return
        try:
            cv2.imwrite(path, img)
            messagebox.showinfo("Guardado", f"Imagen guardada en:\n{path}")
        except Exception as e:
            messagebox.showerror("Error al guardar", str(e))

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
    def download_angles_csv(self):
        """Copia el CSV temporal de ángulos por ROI a la ubicación elegida por el usuario."""
        import shutil
        if self._last_result is None or not self._last_result.angles_csv_path:
            messagebox.showwarning("Sin datos", "Ejecuta el analisis primero.")
            return
        src = self._last_result.angles_csv_path
        if not os.path.isfile(src):
            messagebox.showerror("Archivo no encontrado",
                                 "El CSV temporal ya no existe. Vuelve a ejecutar el analisis.")
            return
        img_stem = os.path.splitext(os.path.basename(self.current_image_path or "resultado"))[0]
        path = filedialog.asksaveasfilename(
            title="Guardar CSV de angulos por ROI",
            initialfile=f"{img_stem}_angulos.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
        )
        if not path:
            return
        try:
            shutil.copy2(src, path)
            messagebox.showinfo("Guardado", f"CSV guardado en:\n{path}")
        except Exception as e:
            messagebox.showerror("Error al guardar", str(e))

    def _export_treeview_csv(self, tv, default_name: str):
        import csv
        cols = tv["columns"]
        rows = [tv.item(iid)["values"] for iid in tv.get_children()]
        if not rows:
            messagebox.showwarning("Sin datos", "No hay datos para exportar. Ejecuta el analisis primero.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar CSV",
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                writer.writerows(rows)
            messagebox.showinfo("Guardado", f"CSV guardado en:\n{path}")
        except Exception as e:
            messagebox.showerror("Error al guardar CSV", str(e))

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
        if self._last_result is None:
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
            self._build_pdf(path, self._last_result)
            messagebox.showinfo("Exportado", f"PDF generado en:\n{path}")
        except Exception as e:
            messagebox.showerror("Error al exportar PDF", str(e))

    def _build_pdf(self, path: str, result):
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph,
            Spacer, HRFlowable, PageBreak,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.lib.units import inch

        # ---- Colores corporativos ----
        C_BLUE_DARK  = colors.HexColor("#1e3a5f")
        C_BLUE_MID   = colors.HexColor("#2d6fa6")
        C_BLUE_LIGHT = colors.HexColor("#dce8f5")
        C_GRAY_ROW   = colors.HexColor("#f4f7fb")
        C_BORDER     = colors.HexColor("#b0bfcf")
        C_TEXT       = colors.HexColor("#1a1a2e")
        C_GREEN      = colors.HexColor("#2e7d32")
        C_WHITE      = colors.white

        PAGE_W, PAGE_H = landscape(letter)
        MARGIN = 0.55 * inch
        USABLE_W = PAGE_W - 2 * MARGIN

        doc = SimpleDocTemplate(
            path,
            pagesize=landscape(letter),
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN, bottomMargin=MARGIN,
            title="SmartCell AI — Reporte de Analisis",
            author="SmartCell AI Analysis Studio",
        )

        styles = getSampleStyleSheet()

        def _style(name, **kw):
            return ParagraphStyle(name, parent=styles["Normal"], **kw)

        st_title   = _style("Title",   fontSize=20, textColor=C_WHITE,      leading=26, fontName="Helvetica-Bold")
        st_sub     = _style("Sub",     fontSize=10, textColor=C_BLUE_LIGHT,  leading=14, fontName="Helvetica")
        st_meta    = _style("Meta",    fontSize=8,  textColor=colors.grey,   leading=11, fontName="Helvetica")
        st_section = _style("Sec",     fontSize=12, textColor=C_BLUE_DARK,   leading=16, fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4)
        st_desc    = _style("Desc",    fontSize=8,  textColor=C_TEXT,        leading=12, fontName="Helvetica",      spaceAfter=6)
        st_caption = _style("Cap",     fontSize=7,  textColor=colors.grey,   leading=10, fontName="Helvetica-Oblique", spaceAfter=10)

        def _hr():
            return HRFlowable(width="100%", thickness=1, color=C_BORDER, spaceAfter=6, spaceBefore=2)

        def _tbl_style(header_bg=C_BLUE_MID, row_alt=C_GRAY_ROW):
            return TableStyle([
                # Encabezado
                ("BACKGROUND",    (0, 0), (-1, 0),  header_bg),
                ("TEXTCOLOR",     (0, 0), (-1, 0),  C_WHITE),
                ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, 0),  8),
                ("TOPPADDING",    (0, 0), (-1, 0),  5),
                ("BOTTOMPADDING", (0, 0), (-1, 0),  5),
                ("ALIGN",         (0, 0), (-1, 0),  "CENTER"),
                # Filas de datos
                ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",      (0, 1), (-1, -1), 7),
                ("TOPPADDING",    (0, 1), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
                ("ALIGN",         (0, 1), (-1, -1), "CENTER"),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                # Bordes
                ("GRID",          (0, 0), (-1, -1), 0.4, C_BORDER),
                ("LINEBELOW",     (0, 0), (-1, 0),  1.2, C_BLUE_DARK),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, row_alt]),
            ])

        story = []

        # =====================================================================
        # PORTADA / CABECERA
        # =====================================================================
        now_str = datetime.datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
        img_name = os.path.basename(self.current_image_path) if self.current_image_path else "—"

        header_data = [[
            Paragraph("SmartCell AI Analysis Studio", st_title),
            Paragraph(f"Reporte de Análisis Celular<br/>"
                      f"<font size='9'>{now_str}</font>", st_sub),
        ]]
        header_tbl = Table(header_data, colWidths=[USABLE_W * 0.62, USABLE_W * 0.38])
        header_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_BLUE_DARK),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",(0, 0), (0, 0),   16),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING",  (1, 0), (1, 0),   14),
            ("ALIGN",      (1, 0), (1, 0),   "RIGHT"),
        ]))
        story.append(header_tbl)
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Imagen analizada: {img_name}", st_meta))
        story.append(_hr())

        # =====================================================================
        # SECCIÓN 1 — RESUMEN
        # =====================================================================
        story.append(Paragraph("1.  Resumen Ejecutivo", st_section))
        story.append(Paragraph(
            "El siguiente cuadro presenta los indicadores estadísticos globales obtenidos "
            "tras el análisis de segmentación celular realizado con el modelo Cellpose "
            "y la CNN de orientación de fibras.",
            st_desc,
        ))

        summary_data = [
            ["Indicador", "Valor", "Descripción"],
            ["Total de células detectadas", str(result.count),
             "Número de objetos segmentados por Cellpose"],
            ["Área media", f"{result.mean_area:.1f} px²",
             "Promedio del área (en píxeles) de todas las células"],
            ["Desviación estándar del área", f"{result.std_area:.1f} px²",
             "Dispersión del tamaño celular respecto a la media"],
        ]
        col_w = [USABLE_W * 0.32, USABLE_W * 0.18, USABLE_W * 0.50]
        summary_tbl = Table(summary_data, colWidths=col_w, repeatRows=1, splitByRow=True)
        summary_tbl.setStyle(TableStyle([
            *_tbl_style(header_bg=C_BLUE_DARK)._cmds,
            ("ALIGN", (0, 1), (0, -1), "LEFT"),
            ("ALIGN", (2, 1), (2, -1), "LEFT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(summary_tbl)
        story.append(Spacer(1, 10))

        # =====================================================================
        # SECCIÓN 2 — ORIENTACIÓN
        # =====================================================================
        story.append(_hr())
        story.append(Paragraph("2.  Orientación por Célula", st_section))
        story.append(Paragraph(
            "La columna <b>CNN (°)</b> contiene el ángulo estimado por la red neuronal "
            "convolucional de orientación (rango 0° – 180°). "
            "La columna <b>Feature (°)</b> corresponde al ángulo del eje mayor de la "
            "elipse ajustada a la máscara binaria (rango −90° – 90°). "
            "La columna <b>Método</b> indica si se usó la CNN o el fallback geométrico.",
            st_desc,
        ))

        orient_header = ["#", "CNN (°)", "Feature (°)", "Método"]
        orient_rows = [
            [str(i), f"{a:.1f}", f"{f:.1f}", "Elipse" if fb else "CNN"]
            for i, (a, f, fb) in enumerate(
                zip(result.angles, result.feature_angles, result.used_fallback), 1
            )
        ]
        n_orient = len(orient_rows)
        orient_data = [orient_header] + orient_rows
        col_w_o = [USABLE_W * 0.07, USABLE_W * 0.15, USABLE_W * 0.17, USABLE_W * 0.14]
        orient_tbl = Table(orient_data, colWidths=col_w_o, repeatRows=1, splitByRow=True)
        orient_tbl.setStyle(_tbl_style())
        story.append(orient_tbl)
        story.append(Paragraph(f"Total: {n_orient} células", st_caption))

        # =====================================================================
        # SECCIÓN 3 — DETALLE
        # =====================================================================
        story.append(_hr())
        story.append(Paragraph("3.  Detalle por Célula", st_section))
        story.append(Paragraph(
            "Combina el área en píxeles con los ángulos de orientación CNN y geométrico "
            "para facilitar la comparación individual de cada célula detectada.",
            st_desc,
        ))

        detail_header = ["#", "Área (px²)", "CNN (°)", "Feature (°)", "Método"]
        detail_rows = [
            [str(i), f"{area:.0f}", f"{a:.1f}", f"{f:.1f}", "Elipse" if fb else "CNN"]
            for i, (area, a, f, fb) in enumerate(
                zip(result.areas, result.angles, result.feature_angles, result.used_fallback), 1
            )
        ]
        detail_data = [detail_header] + detail_rows
        col_w_d = [USABLE_W * 0.06, USABLE_W * 0.13, USABLE_W * 0.13, USABLE_W * 0.15, USABLE_W * 0.13]
        detail_tbl = Table(detail_data, colWidths=col_w_d, repeatRows=1, splitByRow=True)
        detail_tbl.setStyle(_tbl_style(header_bg=colors.HexColor("#1b5e20"), row_alt=colors.HexColor("#f1f8f1")))
        story.append(detail_tbl)
        story.append(Paragraph(f"Total: {len(detail_rows)} células", st_caption))

        # =====================================================================
        # SECCIÓN 4 — FEATURES MORFOLÓGICAS (página nueva)
        # =====================================================================
        story.append(PageBreak())
        story.append(Paragraph("4.  Features Morfológicas", st_section))
        story.append(Paragraph(
            "Métricas calculadas sobre el <i>label map</i> de Cellpose mediante "
            "<b>skimage.measure.regionprops</b>. "
            "El <b>área</b> y <b>perímetro</b> se expresan en píxeles. "
            "La <b>excentricidad</b> varía entre 0 (círculo perfecto) y 1 (elipse muy alargada). "
            "La <b>solidez</b> es la relación entre el área de la célula y su casco convexo "
            "(1 = forma convexa perfecta). "
            "La <b>orientación</b> es el ángulo del eje mayor en grados respecto al eje horizontal. "
            "Los <b>centroides</b> indican la posición del centro de masa en píxeles (x, y).",
            st_desc,
        ))

        feat_header = ["#", "Área", "Perímetro", "Excentr.", "Solidez",
                       "Eje Mayor", "Eje Menor", "Orient. (°)",
                       "Circularidad", "Diám. Feret",
                       "Centroide X", "Centroide Y"]
        feat_rows = [
            [
                str(f["label"]),
                str(f["area"]),
                f"{f['perimeter']:.2f}",
                f"{f['eccentricity']:.4f}",
                f"{f['solidity']:.4f}",
                f"{f['major_axis']:.2f}",
                f"{f['minor_axis']:.2f}",
                f"{f['orientation']:.2f}",
                f"{f['circularity']:.4f}",
                f"{f['feret_diameter']:.4f}",
                f"{f['centroid_x']:.2f}",
                f"{f['centroid_y']:.2f}",
            ]
            for f in result.cell_features
        ]
        feat_data = [feat_header] + feat_rows
        col_w_f = [USABLE_W / 12] * 12
        feat_tbl = Table(feat_data, colWidths=col_w_f, repeatRows=1, splitByRow=True)
        feat_tbl.setStyle(_tbl_style(
            header_bg=colors.HexColor("#4a148c"),
            row_alt=colors.HexColor("#f5f0fb"),
        ))
        story.append(feat_tbl)
        story.append(Paragraph(f"Total: {len(feat_rows)} células", st_caption))

        # =====================================================================
        # PIE DE PÁGINA via onPage callback
        # =====================================================================
        def _add_footer(canvas_obj, doc_obj):
            canvas_obj.saveState()
            canvas_obj.setFont("Helvetica", 7)
            canvas_obj.setFillColor(colors.grey)
            canvas_obj.drawString(MARGIN, 0.35 * inch,
                                  "SmartCell AI Analysis Studio  —  Reporte generado automaticamente")
            canvas_obj.drawRightString(
                PAGE_W - MARGIN, 0.35 * inch,
                f"Pagina {doc_obj.page}  |  {now_str}",
            )
            canvas_obj.restoreState()

        doc.build(story, onFirstPage=_add_footer, onLaterPages=_add_footer)

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

    # =========================================================================
    # Descarga automática de modelos
    # =========================================================================
    def _check_and_download_models(self):
        """Verifica si los modelos existen; si no, ofrece descargarlos."""
        import glob as _glob
        missing = not (_glob.glob(_DETECTION_MODEL + "*") or os.path.isfile(_DETECTION_MODEL))
        if not missing:
            return

        answer = messagebox.askyesno(
            "Modelos no encontrados",
            "El modelo de detección Cellpose no está descargado.\n\n"
            "¿Deseas descargarlo ahora desde Google Drive?\n"
            "(~1.2 GB — puede tardar varios minutos según tu conexión)",
        )
        if not answer:
            self._set_status("Modelo no disponible — descarga pendiente")
            return

        self._run_model_download()

    def _run_model_download(self):
        """Muestra un diálogo con progreso real y descarga los modelos en background."""
        dlg = tk.Toplevel(self)
        dlg.title("Descargando modelo…")
        dlg.geometry("460x200")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.protocol("WM_DELETE_WINDOW", lambda: None)  # bloquear cierre manual

        tb.Label(dlg, text="Descargando modelo Cellpose",
                 font=("Segoe UI", 11, "bold")).pack(pady=(18, 2))
        tb.Label(dlg, text="Esto solo ocurre una vez. No cierres la ventana.",
                 bootstyle="secondary", font=("Segoe UI", 9)).pack()

        pb_var = tk.DoubleVar(value=0)
        pb = tb.Progressbar(dlg, variable=pb_var, maximum=100,
                            bootstyle="primary", length=380)
        pb.pack(pady=(12, 4))

        pct_var  = tk.StringVar(value="0 %")
        size_var = tk.StringVar(value="")
        tb.Label(dlg, textvariable=pct_var,
                 font=("Segoe UI", 10, "bold"), bootstyle="primary").pack()
        tb.Label(dlg, textvariable=size_var,
                 font=("Segoe UI", 8), bootstyle="secondary").pack(pady=(2, 0))

        # Compartir estado entre threads via dict mutable
        _state = {"total": 0, "downloaded": 0, "done": False, "ok": False, "err": ""}

        def _poll_progress():
            """Actualiza la barra cada 400 ms leyendo el tamaño del archivo parcial."""
            if _state["done"]:
                return
            dest = os.path.join(_PROJECT_ROOT, "Models", "model_base_3B")
            if os.path.isfile(dest):
                downloaded = os.path.getsize(dest)
                _state["downloaded"] = downloaded
                total = _state["total"] or 1_220_000_000  # fallback 1.22 GB
                pct = min(downloaded / total * 100, 99)
                pb_var.set(pct)
                pct_var.set(f"{pct:.1f} %")
                mb_dl = downloaded / 1_048_576
                mb_tot = total / 1_048_576
                size_var.set(f"{mb_dl:.0f} MB  /  {mb_tot:.0f} MB")
            self.after(400, _poll_progress)

        def worker():
            try:
                import gdown
                file_id = "1rJzPz5gvGkDMWkkba7f81Y5hVr6yeqkd"
                url = f"https://drive.google.com/file/d/{file_id}/view?usp=drive_link"
                dest = os.path.join(_PROJECT_ROOT, "Models", "model_base_3B")
                result = gdown.download(url, output=dest, fuzzy=True,
                                        quiet=True, resume=True)
                _state["done"] = True
                _state["ok"] = result is not None
                if not _state["ok"]:
                    _state["err"] = "gdown retornó None — archivo privado o ID incorrecto."
            except Exception as e:
                _state["done"] = True
                _state["ok"] = False
                _state["err"] = str(e)
            self.after(0, _on_done)

        def _on_done():
            pb_var.set(100)
            dlg.destroy()
            if _state["ok"]:
                messagebox.showinfo("Descarga completa",
                                    "El modelo se descargó correctamente.\n"
                                    "Ya puedes usar el botón Analizar.")
                self._set_status("Modelo listo")
            else:
                messagebox.showerror(
                    "Error de descarga",
                    f"No se pudo descargar el modelo.\n\n{_state['err']}\n\n"
                    "Alternativamente descárgalo manualmente desde:\n"
                    "https://drive.google.com/file/d/1rJzPz5gvGkDMWkkba7f81Y5hVr6yeqkd/view\n"
                    "y colócalo en la carpeta models/ con el nombre model_base_3B",
                )
                self._set_status("Modelo no disponible")

        self.after(400, _poll_progress)
        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    app = App()
    app.after(500, app._check_and_download_models)
    app.mainloop()
