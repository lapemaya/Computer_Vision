"""
ViewGallery.py - Galleria per visualizzare le immagini croppate salvate nel database
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from pathlib import Path
from dataset import ObjectDatabase
import math


class ImageGallery:
    def __init__(self, root):
        self.root = root
        self.root.title("📸 Galleria Database Oggetti")
        self.root.geometry("1200x800")

        # Inizializza database
        self.db = ObjectDatabase(db_path="detections.db", feature_dir="features_db", crops_dir="crops_db")

        # Variabili
        self.current_class = None
        self.images_data = []
        self.thumb_size = (150, 150)

        # Setup UI
        self.setup_ui()

        # Carica classi
        self.load_classes()

    def setup_ui(self):
        """Configura l'interfaccia utente"""
        # Frame principale
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configura grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Header con statistiche
        header_frame = ttk.LabelFrame(main_frame, text="📊 Statistiche Database", padding="10")
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.stats_label = ttk.Label(header_frame, text="Caricamento...", font=("Arial", 10))
        self.stats_label.pack()

        # Frame selezione classe
        class_frame = ttk.LabelFrame(main_frame, text="🗂️ Seleziona Classe", padding="10")
        class_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Dropdown classi
        ttk.Label(class_frame, text="Classe:").grid(row=0, column=0, padx=(0, 10))

        self.class_var = tk.StringVar()
        self.class_dropdown = ttk.Combobox(class_frame, textvariable=self.class_var, state="readonly", width=30)
        self.class_dropdown.grid(row=0, column=1, padx=(0, 10))
        self.class_dropdown.bind("<<ComboboxSelected>>", self.on_class_selected)

        # Bottone mostra tutti
        ttk.Button(class_frame, text="📋 Mostra Tutti", command=self.show_all).grid(row=0, column=2, padx=(0, 10))

        # Bottone refresh
        ttk.Button(class_frame, text="🔄 Aggiorna", command=self.load_classes).grid(row=0, column=3)

        # Info classe corrente
        self.class_info_label = ttk.Label(class_frame, text="", font=("Arial", 9, "italic"))
        self.class_info_label.grid(row=1, column=0, columnspan=4, pady=(5, 0))

        # Frame galleria con scrollbar
        gallery_frame = ttk.LabelFrame(main_frame, text="🖼️ Galleria Immagini", padding="10")
        gallery_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        gallery_frame.columnconfigure(0, weight=1)
        gallery_frame.rowconfigure(0, weight=1)

        # Canvas per scrolling
        self.canvas = tk.Canvas(gallery_frame, bg="white")
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbars
        vsb = ttk.Scrollbar(gallery_frame, orient="vertical", command=self.canvas.yview)
        vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))

        hsb = ttk.Scrollbar(gallery_frame, orient="horizontal", command=self.canvas.xview)
        hsb.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Frame interno per le immagini
        self.gallery_inner_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.gallery_inner_frame, anchor="nw")

        # Bind resize
        self.gallery_inner_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # Bind mouse wheel
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)

        # Footer con controlli
        footer_frame = ttk.Frame(main_frame)
        footer_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Button(footer_frame, text="❌ Chiudi", command=self.root.quit).pack(side=tk.RIGHT)
        ttk.Button(footer_frame, text="📤 Esporta JSON", command=self.export_json).pack(side=tk.RIGHT, padx=(0, 10))

    def on_frame_configure(self, event=None):
        """Aggiorna la scrollregion quando il frame cambia dimensione"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """Ridimensiona il frame interno quando il canvas cambia dimensione"""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    def on_mousewheel(self, event):
        """Gestisce lo scroll con la rotella del mouse"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_classes(self):
        """Carica le classi dal database"""
        classes = self.db.get_all_classes()

        if not classes:
            messagebox.showinfo("Info", "Nessuna classe trovata nel database.")
            self.stats_label.config(text="Database vuoto")
            return

        # Aggiorna dropdown
        class_names = [cls['class_name'] for cls in classes]
        self.class_dropdown['values'] = class_names

        # Calcola statistiche
        total_objects = sum(cls['object_count'] for cls in classes)
        stats_text = f"📦 Totale Classi: {len(classes)} | 🎯 Totale Oggetti: {total_objects}"
        self.stats_label.config(text=stats_text)

        # Seleziona la prima classe di default
        if class_names:
            self.class_var.set(class_names[0])
            self.on_class_selected()

    def on_class_selected(self, event=None):
        """Gestisce la selezione di una classe"""
        class_name = self.class_var.get()
        if class_name:
            self.show_class(class_name)

    def show_class(self, class_name):
        """Mostra le immagini di una specifica classe"""
        self.current_class = class_name

        # Recupera oggetti della classe
        objects = self.db.get_class_objects(class_name)

        if not objects:
            self.class_info_label.config(text=f"Nessun oggetto trovato per la classe '{class_name}'")
            self.clear_gallery()
            return

        # Filtra oggetti con immagini esistenti
        valid_objects = []
        for obj in objects:
            if obj['crop_image_path'] and Path(obj['crop_image_path']).exists():
                valid_objects.append(obj)

        if not valid_objects:
            self.class_info_label.config(text=f"Nessuna immagine disponibile per '{class_name}' ({len(objects)} oggetti senza immagine)")
            self.clear_gallery()
            return

        # Aggiorna info
        info_text = f"Classe: {class_name} | Oggetti: {len(valid_objects)}/{len(objects)}"
        self.class_info_label.config(text=info_text)

        # Mostra galleria
        self.display_gallery(valid_objects)

    def show_all(self):
        """Mostra tutte le immagini di tutte le classi"""
        self.current_class = None

        classes = self.db.get_all_classes()
        all_objects = []

        for cls in classes:
            objects = self.db.get_class_objects(cls['class_name'])
            for obj in objects:
                if obj['crop_image_path'] and Path(obj['crop_image_path']).exists():
                    obj['display_class'] = cls['class_name']
                    all_objects.append(obj)

        if not all_objects:
            self.class_info_label.config(text="Nessuna immagine disponibile nel database")
            self.clear_gallery()
            return

        self.class_info_label.config(text=f"Mostrando tutte le immagini: {len(all_objects)} oggetti")
        self.display_gallery(all_objects)

    def display_gallery(self, objects):
        """Mostra la galleria di immagini"""
        self.clear_gallery()

        # Calcola numero di colonne in base alla larghezza
        canvas_width = self.canvas.winfo_width()
        if canvas_width < 100:
            canvas_width = 1000  # Default width

        cols = max(1, (canvas_width - 20) // (self.thumb_size[0] + 20))

        # Crea griglia di immagini
        for idx, obj in enumerate(objects):
            row = idx // cols
            col = idx % cols

            self.create_image_card(obj, row, col)

        # Aggiorna scrollregion
        self.on_frame_configure()

    def create_image_card(self, obj, row, col):
        """Crea una card per un'immagine"""
        # Frame per la card
        card_frame = ttk.Frame(self.gallery_inner_frame, relief="raised", borderwidth=2)
        card_frame.grid(row=row, column=col, padx=10, pady=10, sticky="n")

        # Carica e ridimensiona immagine
        try:
            img = Image.open(obj['crop_image_path'])
            img.thumbnail(self.thumb_size, Image.Resampling.LANCZOS)

            # Crea immagine Tkinter
            photo = ImageTk.PhotoImage(img)

            # Label immagine
            img_label = ttk.Label(card_frame, image=photo)
            img_label.image = photo  # Mantieni riferimento
            img_label.pack()

            # Info oggetto
            info_text = f"ID: {obj['object_id']}"
            if 'display_class' in obj:
                info_text += f"\nClasse: {obj['display_class']}"
            if obj.get('confidence'):
                info_text += f"\nConf: {obj['confidence']:.2f}"
            info_text += f"\nData: {obj['detection_date']}"

            info_label = ttk.Label(card_frame, text=info_text, font=("Arial", 8), justify="center")
            info_label.pack(pady=(5, 0))

            # Bind click per dettagli
            img_label.bind("<Button-1>", lambda e, o=obj: self.show_details(o))

        except Exception as e:
            error_label = ttk.Label(card_frame, text=f"Errore caricamento:\n{str(e)[:30]}",
                                   font=("Arial", 8), foreground="red")
            error_label.pack()

    def show_details(self, obj):
        """Mostra i dettagli di un oggetto in una finestra popup"""
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Dettagli Oggetto #{obj['object_id']}")
        detail_window.geometry("600x700")

        # Frame principale
        main_frame = ttk.Frame(detail_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Immagine grande
        try:
            img = Image.open(obj['crop_image_path'])
            # Ridimensiona mantenendo aspect ratio
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            img_label = ttk.Label(main_frame, image=photo)
            img_label.image = photo
            img_label.pack(pady=(0, 20))
        except Exception as e:
            ttk.Label(main_frame, text=f"Errore caricamento immagine: {e}").pack()

        # Dettagli testuali
        details_frame = ttk.LabelFrame(main_frame, text="📋 Informazioni", padding="10")
        details_frame.pack(fill=tk.BOTH, expand=True)

        details = [
            ("ID Oggetto", obj['object_id']),
            ("Classe", obj.get('display_class', self.current_class)),
            ("Confidence", f"{obj.get('confidence', 'N/A'):.3f}" if obj.get('confidence') else "N/A"),
            ("Data Rilevamento", obj['detection_date']),
            ("Ora Rilevamento", obj['detection_time']),
            ("Path Immagine", obj.get('crop_image_path', 'N/A')),
            ("Path Feature", obj.get('feature_path', 'N/A')),
        ]

        if obj.get('bbox'):
            bbox = obj['bbox']
            bbox_str = f"x1:{bbox.get('x1','?')}, y1:{bbox.get('y1','?')}, x2:{bbox.get('x2','?')}, y2:{bbox.get('y2','?')}"
            details.append(("Bounding Box", bbox_str))

        for i, (label, value) in enumerate(details):
            ttk.Label(details_frame, text=f"{label}:", font=("Arial", 9, "bold")).grid(row=i, column=0, sticky="w", pady=2, padx=(0, 10))
            ttk.Label(details_frame, text=str(value), font=("Arial", 9)).grid(row=i, column=1, sticky="w", pady=2)

        # Bottone chiudi
        ttk.Button(main_frame, text="Chiudi", command=detail_window.destroy).pack(pady=(10, 0))

    def clear_gallery(self):
        """Pulisce la galleria"""
        for widget in self.gallery_inner_frame.winfo_children():
            widget.destroy()
        self.canvas.yview_moveto(0)

    def export_json(self):
        """Esporta il database in JSON"""
        output_path = "database_export.json"
        try:
            self.db.export_to_json(output_path)
            messagebox.showinfo("Successo", f"Database esportato in:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante l'esportazione:\n{str(e)}")

    def __del__(self):
        """Chiude il database quando la finestra viene chiusa"""
        if hasattr(self, 'db'):
            self.db.close()


def main():
    """Funzione principale"""
    root = tk.Tk()
    app = ImageGallery(root)
    root.mainloop()


if __name__ == "__main__":
    main()

