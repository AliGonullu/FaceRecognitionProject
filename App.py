import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import Basics
import DetectFaces
import cv2
import numpy as np

class FaceRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("900x700")
        self.root.configure(bg="#e9ecef")

        self.database_dir = "database"
        self.images_dir = "Images"
        os.makedirs(self.database_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure('TFrame', background='#f8f9fa')
        self.style.configure('TLabel', background='#f8f9fa', font=('Segoe UI', 10))
        self.style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'), foreground="#343a40", background="#f8f9fa")
        self.style.configure('TButton', font=('Segoe UI', 10), padding=6, background='#007bff', foreground='white')
        self.style.map("TButton",
                       foreground=[("pressed", "white"), ("active", "white")],
                       background=[("pressed", "#0056b3"), ("active", "#0069d9")])
        self.style.configure('TCombobox', padding=5)

        self.create_widgets()
        self.setup_image_list()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root, padding=20, relief=tk.GROOVE, borderwidth=2)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(self.main_frame, text="Face Recognition System", style='Title.TLabel').grid(
            row=0, column=0, columnspan=4, pady=(0, 20))

        ttk.Label(self.main_frame, text="Select Base Image:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.image_combobox = ttk.Combobox(self.main_frame, state="readonly", width=45)
        self.image_combobox.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        self.compare_btn = ttk.Button(self.main_frame, text="Start", command=self.compare_faces)
        self.compare_btn.grid(row=1, column=2, padx=5, pady=5)

        ttk.Separator(self.main_frame, orient='horizontal').grid(row=2, column=0, columnspan=4, sticky="ew", pady=15)
        ttk.Label(self.main_frame, text="Add New Person:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.add_image_btn = ttk.Button(self.main_frame, text="Add Image", command=self.add_new_image)
        self.add_image_btn.grid(row=3, column=2, padx=5, pady=5)

        self.remove_image_btn = ttk.Button(self.main_frame, text="Remove Image", command=self.remove_image)
        self.remove_image_btn.grid(row=4, column=2, padx=5, pady=(5, 15))

        self.image_frame = ttk.Frame(self.main_frame, relief=tk.SUNKEN, borderwidth=1)
        self.image_frame.grid(row=5, column=0, columnspan=4, sticky=tk.NSEW, pady=10)

        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(5, weight=1)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor=tk.W, background="#dee2e6", font=('Segoe UI', 9)).pack(fill=tk.X, side=tk.BOTTOM)
    
    def setup_image_list(self):
        DetectFaces.Image_Paths = self.get_all_image_paths()
        
        image_names = []
        for path in DetectFaces.Image_Paths:
            filename = os.path.basename(path)
            person_name = os.path.basename(os.path.dirname(path))
            image_names.append(f"{person_name} - {filename}")
        
        self.image_combobox['values'] = image_names
        if image_names:
            self.image_combobox.current(0)
    
    def get_all_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.images_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    

    def add_new_image(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Select Face Image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png")],
                initialdir=os.path.expanduser("~"))
            
            if not file_path: return

            with open(file_path, 'rb') as f:
                img_data = f.read()
            
            nparr = np.frombuffer(img_data, np.uint8)
            test_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if test_img is None:
                raise ValueError("OpenCV cannot read this image file")
            
            h, w = test_img.shape[:2]
            min_size = 575
            max_size = 1080

            if not (min_size <= w <= max_size and min_size <= h <= max_size):
                error_message = (
                    f"Image size is outside the allowed range.\n"
                    f"Current size: {w}x{h}\n"
                    f"Required size: Minimum {min_size}x{min_size}, Maximum {max_size}x{max_size}."
                )
                messagebox.showerror("Invalid Image Size", error_message)
                self.status_var.set("Error: Image size invalid")
                return
            
            safe_name = "".join(c for c in os.path.basename(file_path) 
                            if c.isalnum() or c in (' ', '.', '_')).rstrip()
            dest_path = os.path.join(self.images_dir, safe_name)
            
            with open(dest_path, 'wb') as f:
                f.write(img_data)

            DetectFaces.Image_Paths = self.get_all_image_paths()
            self.setup_image_list()
            self.status_var.set(f"Added: {safe_name}")
            
        except Exception as e:
            messagebox.showerror("Error", 
                f"Failed to add image:\n{str(e)}\n"
                f"Try: 1. Saving locally first\n"
                f"2. Using simpler filename\n"
                f"3. Different image format")
            self.status_var.set("Error adding image")


    def remove_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")],
            initialdir=os.path.abspath(self.images_dir))
        
        if not file_path: return
            
        try:
            os.remove(file_path)
            self.setup_image_list()
            messagebox.showinfo("Success", "Image removed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove image: {str(e)}")
            self.status_var.set("Error adding image")
    
    def compare_faces(self):
        selected_idx = self.image_combobox.current()
        if selected_idx < 0:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        self.status_var.set("Processing face comparison...")
        self.root.update()
        
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        Basics.CompareImage(selected_idx)
        self.status_var.set("Comparison complete")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognizerGUI(root)
    root.mainloop()