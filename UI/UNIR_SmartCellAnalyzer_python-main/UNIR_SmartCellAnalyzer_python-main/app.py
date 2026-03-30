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

# NUEVO: Librerías para PDF
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ---------------------------------------------------------------------------
# Configurar sys.path para encontrar Detection/ y Core/ desde la raíz del proyecto
# app.py está en UI/UNIR_.../UNIR_.../app.py → subir 3 niveles para llegar a la raíz
# ---------------------------------------------------------------------------
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_APP_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Rutas a los modelos
_DETECTION_MODEL = os.path.join(_PROJECT_ROOT, "Models", "model_base_3A")
_ORIENTATION_MODEL = os.path.join(
    _PROJECT_ROOT, "Orientation", "Orientador_De_Fibras_CNN", "models", "cnn_fiber_orientation.pth"
)

APP_TITLE = "Smart Cell AI Analysis Studio (Beta)"


class App(tb.Window):
    def __init__(self):
        super().__init__(themename="flatly")

        self.title(APP_TITLE)
        self.geometry("1180x690")
        self.minsize(980, 560)

        self.current_image = None
        self.current_image_tk = None
        self.current_image_path = None
        self._analyze_btn = None  # referencia al botón para activar/desactivar

        # Estado de zoom y pan
        self._zoom_level = 1.0   # 1.0 = ajuste a canvas; >1 = acercado
        self._pan_x = 0          # desplazamiento horizontal en px de la imagen escalada
        self._pan_y = 0          # desplazamiento vertical
        self._original_image = None  # imagen BGR a resolución completa
        self._drag_start = None  # punto de inicio del arrastre
        self._zoom_label = None  # etiqueta "100 %"

        self.root = tb.Frame(self, padding=10)
        self.root.pack(fill=BOTH, expand=True)

        self._create_menu()
        self._create_body()

    # =========================
    # Menu
    # =========================
    def _create_menu(self):
        menubar = tk.Menu(self)

        menu_archivo = tk.Menu(menubar, tearoff=0)
        menu_archivo.add_command(label="Abrir imagen…", command=self.open_image)
        menu_archivo.add_separator()
        menu_archivo.add_command(label="Salir", command=self.destroy)

        menu_exportar = tk.Menu(menubar, tearoff=0)
        menu_exportar.add_command(label="Exportar a PDF…", command=self.export_pdf)

        menubar.add_cascade(label="Archivo", menu=menu_archivo)
        menubar.add_cascade(label="Exportar", menu=menu_exportar)

        self.config(menu=menubar)

    # =========================
    # Main body
    # =========================
    def _create_body(self):
        body = tb.Frame(self.root)
        body.pack(fill=BOTH, expand=True)

        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # ---- Left panel ----
        left_card = tb.Frame(body, bootstyle="light", padding=10)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        left_inner = tb.Frame(left_card, bootstyle="secondary", padding=2)
        left_inner.pack(fill=BOTH, expand=True)

        self.canvas = tk.Canvas(left_inner, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)

        # Bindings para zoom con rueda y arrastre para pan
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        zoom_bar = tb.Frame(left_card)
        zoom_bar.pack(fill=X, pady=(10, 0))

        tb.Label(zoom_bar, text="Zoom:", bootstyle="secondary").pack(side=LEFT)
        tb.Button(zoom_bar, text="+", width=3, bootstyle="secondary", command=self.zoom_in).pack(side=LEFT, padx=(8, 6))
        tb.Button(zoom_bar, text="−", width=3, bootstyle="secondary", command=self.zoom_out).pack(side=LEFT, padx=(0, 8))
        tb.Button(zoom_bar, text="[ ]", width=4, bootstyle="secondary", command=self.zoom_reset).pack(side=LEFT)
        self._zoom_label = tb.Label(zoom_bar, text="Ajuste", bootstyle="secondary", width=7)
        self._zoom_label.pack(side=LEFT, padx=(10, 0))

        self._draw_mock_view()

        # ---- Right panel ----
        right = tb.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")

        analysis_card = tb.Frame(right, bootstyle="light", padding=10)
        analysis_card.pack(fill=BOTH, expand=True)

        header = tb.Frame(analysis_card, bootstyle="primary", padding=(10, 8))
        header.pack(fill=X)

        tb.Label(header, text="Análisis", bootstyle="inverse-primary",
                 font=("Segoe UI", 11, "bold")).pack(anchor=W)

        content = tb.Frame(analysis_card, padding=(10, 10))
        content.pack(fill=BOTH, expand=True)

        self.analysis_text = tk.Text(content, height=10, wrap="word", bd=1, relief="solid")
        self.analysis_text.insert("1.0", "Análisis pendiente de implementar.\nEsperando módulo externo.")
        self.analysis_text.configure(state="disabled")
        self.analysis_text.pack(fill=BOTH, expand=True, pady=(0, 12))

        self._analyze_btn = tb.Button(content, text="Analizar", bootstyle="primary",
                  command=self.analyze)
        self._analyze_btn.pack(fill=X, ipady=6, pady=(0, 10))

        tb.Button(content, text="Guardar Análisis", bootstyle="secondary",
                  command=self.save_analysis).pack(fill=X, ipady=6, pady=(0, 10))

        tb.Button(content, text="Exportar a PDF", bootstyle="secondary",
                  command=self.export_pdf).pack(fill=X, ipady=6)

    # =========================
    # Mock view
    # =========================
    def _draw_mock_view(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 760
        h = self.canvas.winfo_height() or 480

        self.canvas.create_rectangle(20, 20, w - 20, h - 20,
                                     fill="#5b6f86", outline="#3a3a3a", width=2)

    # =========================
    # Abrir imagen real
    # =========================
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Abrir imagen",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
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

        self.display_image(self.current_image)

    # =========================
    # Mostrar imagen en canvas
    # =========================
    def display_image(self, img_cv):
        """Carga una nueva imagen: resetea zoom y pan, luego renderiza."""
        self._original_image = img_cv.copy()
        self._zoom_level = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._render_image()

    def _render_image(self):
        """Renderiza _original_image con el zoom y pan actuales."""
        if self._original_image is None:
            return

        img = self._original_image
        H, W = img.shape[:2]
        cw = self.canvas.winfo_width() or 760
        ch = self.canvas.winfo_height() or 480

        # Escala base para ajustar la imagen al canvas (fit)
        fit_scale = min(cw / W, ch / H)
        scale = fit_scale * self._zoom_level

        new_w = max(1, int(W * scale))
        new_h = max(1, int(H * scale))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((new_w, new_h), Image.LANCZOS)

        # Fondo negro del canvas
        canvas_img = Image.new("RGB", (cw, ch), (43, 43, 43))

        if new_w <= cw and new_h <= ch:
            # Imagen cabe en el canvas: centrarla, sin pan
            self._pan_x = 0
            self._pan_y = 0
            x0 = (cw - new_w) // 2
            y0 = (ch - new_h) // 2
            canvas_img.paste(img_pil, (x0, y0))
        else:
            # Imagen más grande que el canvas: recortar con pan
            # Centro de la vista en coordenadas de la imagen escalada
            cx = new_w // 2 + self._pan_x
            cy = new_h // 2 + self._pan_y

            # Limitar para no salir de los bordes
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
        if self._zoom_label is None:
            return
        if self._original_image is None:
            self._zoom_label.configure(text="Ajuste")
            return
        H, W = self._original_image.shape[:2]
        cw = self.canvas.winfo_width() or 760
        ch = self.canvas.winfo_height() or 480
        fit_scale = min(cw / W, ch / H)
        pct = int(fit_scale * self._zoom_level * 100)
        self._zoom_label.configure(text=f"{pct} %")

    # =========================
    # Ejecutar análisis (pipeline real)
    # =========================
    def analyze(self):
        if self.current_image is None or self.current_image_path is None:
            messagebox.showwarning("Sin imagen", "Primero abre una imagen.")
            return

        if not os.path.exists(_DETECTION_MODEL) and not any(
            os.path.exists(_DETECTION_MODEL + ext) for ext in ["", "_0", "_1"]
        ):
            # Verificar con glob por si tiene sufijo de fecha
            import glob
            candidates = glob.glob(_DETECTION_MODEL + "*")
            if not candidates:
                messagebox.showerror(
                    "Modelo no encontrado",
                    f"No se encontró el modelo de detección en:\n{_DETECTION_MODEL}\n\n"
                    "Coloca el modelo Cellpose en la carpeta Models/."
                )
                return

        self._set_status("Analizando… (esto puede tardar varios segundos)")
        if self._analyze_btn:
            self._analyze_btn.configure(state="disabled")

        def worker():
            try:
                from Core.pipeline import run_analysis, AnalysisResult
                result = run_analysis(
                    self.current_image_path,
                    _DETECTION_MODEL,
                    _ORIENTATION_MODEL,
                )
                self.after(0, lambda: self._on_analysis_complete(result))
            except Exception as e:
                self.after(0, lambda: self._on_analysis_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_analysis_complete(self, result):
        """Actualiza la UI con los resultados del pipeline."""
        # Mostrar figura de 6 paneles en el visor (report_figure es BGR)
        self.current_image = result.report_figure
        self.display_image(result.report_figure)

        # Tabla summary + detalle por célula
        self.analysis_text.configure(state="normal")
        self.analysis_text.delete("1.0", "end")

        # --- Summary ---
        self.analysis_text.insert("end", "=== RESUMEN ===\n")
        self.analysis_text.insert("end", f"{'total_cells':<18} {result.count}\n")
        self.analysis_text.insert("end", f"{'mean_area':<18} {result.mean_area:.1f} px\n")
        self.analysis_text.insert("end", f"{'std_area':<18} {result.std_area:.1f} px\n")

        # --- Detalle por célula ---
        if result.count > 0:
            self.analysis_text.insert("end", "\n=== DETALLE ===\n")
            self.analysis_text.insert(
                "end", f"{'#':<5} {'Area(px)':<12} {'Ang.CNN(°)':<13} {'Ang.Feat(°)':<13} {'Metodo'}\n"
            )
            self.analysis_text.insert("end", "-" * 56 + "\n")
            for i, (area, cnn_a, feat_a, fb) in enumerate(
                zip(result.areas, result.angles, result.feature_angles, result.used_fallback), 1
            ):
                method = "Elipse" if fb else "CNN"
                self.analysis_text.insert(
                    "end", f"{i:<5} {area:<12.0f} {cnn_a:<13.1f} {feat_a:<13.1f} {method}\n"
                )
            if any(result.used_fallback):
                n_fb = sum(result.used_fallback)
                self.analysis_text.insert(
                    "end", f"\n* {n_fb} celula(s) usaron fallback geometrico.\n"
                )

        self.analysis_text.configure(state="disabled")

        self._set_status(f"Listo  —  {result.count} celulas detectadas")
        if self._analyze_btn:
            self._analyze_btn.configure(state="normal")

    def _on_analysis_error(self, message: str):
        """Muestra el error y reactiva los controles."""
        messagebox.showerror("Error en el análisis", message)
        self._set_status("Error durante el análisis")
        if self._analyze_btn:
            self._analyze_btn.configure(state="normal")

    def _set_status(self, text: str):
        """Actualiza el título de la ventana con el estado actual."""
        self.title(f"{APP_TITLE}  —  {text}")

    # =========================
    # Guardar análisis (TXT / CSV)
    # =========================
    def save_analysis(self):
        content = self.analysis_text.get("1.0", "end").strip()

        if not content:
            messagebox.showwarning("Sin contenido", "No hay análisis para guardar.")
            return

        path = filedialog.asksaveasfilename(
            title="Guardar análisis",
            defaultextension=".txt",
            filetypes=[("Archivo de texto", "*.txt"), ("CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )

        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            messagebox.showinfo("Guardado", f"Análisis guardado en:\n{path}")

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar:\n{e}")

    # =========================
    # Exportar análisis a PDF
    # =========================
    def export_pdf(self):
        content = self.analysis_text.get("1.0", "end").strip()

        if not content:
            messagebox.showwarning("Sin contenido", "No hay análisis para exportar.")
            return

        path = filedialog.asksaveasfilename(
            title="Exportar a PDF",
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("Todos los archivos", "*.*")]
        )

        if not path:
            return

        try:
            c = canvas.Canvas(path, pagesize=letter)
            width, height = letter

            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, height - 50, "Análisis de Imagen")

            c.setFont("Helvetica", 10)
            c.drawString(50, height - 70, f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            text_obj = c.beginText(50, height - 100)
            text_obj.setFont("Helvetica", 10)

            for line in content.splitlines():
                text_obj.textLine(line)

            c.drawText(text_obj)
            c.showPage()
            c.save()

            messagebox.showinfo("Exportado", f"PDF generado en:\n{path}")

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo exportar:\n{e}")

    # =========================
    # Zoom y pan
    # =========================
    _ZOOM_STEP = 1.25
    _ZOOM_MAX  = 8.0

    def zoom_in(self):
        if self._original_image is None:
            return
        self._zoom_level = min(self._zoom_level * self._ZOOM_STEP, self._ZOOM_MAX)
        self._render_image()

    def zoom_out(self):
        if self._original_image is None:
            return
        self._zoom_level = max(self._zoom_level / self._ZOOM_STEP, 1.0)
        self._pan_x = 0
        self._pan_y = 0
        self._render_image()

    def zoom_reset(self):
        """Vuelve al ajuste completo (fit)."""
        if self._original_image is None:
            return
        self._zoom_level = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._render_image()

    def _on_mousewheel(self, event):
        if self._original_image is None:
            return
        if event.delta > 0:
            self._zoom_level = min(self._zoom_level * self._ZOOM_STEP, self._ZOOM_MAX)
        else:
            self._zoom_level = max(self._zoom_level / self._ZOOM_STEP, 1.0)
            if self._zoom_level == 1.0:
                self._pan_x = 0
                self._pan_y = 0
        self._render_image()

    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y)

    def _on_drag_move(self, event):
        if self._drag_start is None or self._original_image is None:
            return
        if self._zoom_level <= 1.0:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        # Pan inverso: arrastrar derecha mueve la vista a la derecha
        self._pan_x -= dx
        self._pan_y -= dy
        self._render_image()

    def _on_canvas_resize(self, event):
        """Re-renderiza al cambiar el tamaño de la ventana."""
        if self._original_image is not None:
            self._render_image()
        else:
            self._draw_mock_view()


if __name__ == "__main__":
    app = App()
    app.mainloop()