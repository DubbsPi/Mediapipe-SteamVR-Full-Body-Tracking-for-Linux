import tkinter as tk
from tkinter import ttk
import threading
import time
import json
import os

# Setup path
script_path = os.path.realpath(__file__)
script_directory = os.path.dirname(script_path)

class RealTimeGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VR-Tracker Control Panel")
        self.root.geometry("400x300")

        try:
            self.data = self.load_config()
        except:
            # Default data
            self.data = {"camera": 0, "sliders": [0.0, 0.5, 0.15, 5.0, 0.0]}

        self.slider_data = self.data['sliders']

        # Variables that will be updated in real-time
        self.camera = tk.IntVar(value=self.data['camera'])
        self.toggle_state = tk.BooleanVar(value=False)

        # Slider configurations
        self.slider_configs = [
            {"name": "X Offset", "min": -0.5, "max": 1.5, "default": self.slider_data[0]},
            {"name": "Y Offset", "min": 0, "max": 2.5, "default": self.slider_data[1]},
            {"name": "Z Offset", "min": -0.5, "max": 1.0, "default": self.slider_data[2]},
            {"name": "Smoothing", "min": 1, "max": 10, "default": self.slider_data[3]},
            {"name": "Brightness", "min": -50, "max": 50, "default": self.slider_data[4]}
        ]

        # Store slider variables
        self.slider_vars = []
        self.slider_widgets = []

        # Thread control
        self.running = True
        self.background_thread = None

        self.setup_ui()
        self.start_background_thread()

        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Input boxs at the top
        ttk.Label(main_frame, text="Camera:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        camera_entry = ttk.Entry(main_frame, textvariable=self.camera, width=15)
        camera_entry.grid(row=0, column=1, sticky=tk.W, pady=(0, 10))

        # Create sliders dynamically
        for i, config in enumerate(self.slider_configs):
            # Create variable for this slider
            var = tk.DoubleVar(value=config["default"])
            self.slider_vars.append(var)

            # Label
            ttk.Label(main_frame, text=f"{config['name']}:").grid(
                row=i+2, column=0, sticky=tk.W, pady=2
            )

            # Slider
            slider = ttk.Scale(
                main_frame,
                from_=config["min"],
                to=config["max"],
                variable=var,
                orient=tk.HORIZONTAL,
                length=200
            )
            slider.grid(row=i+2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)
            self.slider_widgets.append(slider)

            # Value label
            value_label = ttk.Label(main_frame, text=f"{config['default']:.1f}")
            value_label.grid(row=i+2, column=2, sticky=tk.W, padx=(10, 0), pady=2)

            # Update label when slider changes
            var.trace('w', lambda *args, lbl=value_label, v=var:
                     lbl.config(text=f"{v.get():.1f}"))

        # Toggle button
        toggle_btn = ttk.Checkbutton(
            main_frame,
            text="Pause Tracking",
            variable=self.toggle_state
        )
        toggle_btn.grid(row=len(self.slider_configs)+2, column=0, columnspan=2,
                       sticky=tk.W, pady=(20, 0))

        save_btn = tk.Button(main_frame, text="Save Config", command=self.save_config)
        save_btn.grid(row=len(self.slider_configs)+4, column=0, columnspan=2,
                       sticky=tk.W, pady=(20, 0))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def start_background_thread(self):
        # Start the background thread for real-time processing
        self.background_thread = threading.Thread(target=self.background_worker, daemon=True)
        self.background_thread.start()

    def background_worker(self):
        # Background thread that processes the values in real-time
        while self.running:
            cycles = 0
            try:
                # Get current values (thread-safe)
                input_val = self.camera.get()
                toggle_val = self.toggle_state.get()
                slider_vals = [var.get() for var in self.slider_vars]

                # Sleep to prevent excessive CPU usage
                time.sleep(0.1)  # Update 10 times per second

            except Exception as e:
                print(f"Background thread error: {e}")
                break

    def get_current_values(self):
        # Get current values
        return {
            'camera': self.camera.get(),
            'sliders': [var.get() for var in self.slider_vars],
            'paused': self.toggle_state.get(),
        }

    def save_config(self):
        try:
            with open(f"{script_directory}/config.json", "w") as json_file:
                self.values = self.get_current_values()
                self.filtered = {self.key: self.value for self.key, self.value in self.values.items() if self.key not in ["paused"]}

                json.dump(self.filtered, json_file, indent=4)
                print(f"Data successfully saved to {json_file}")

        except IOError as e:
            print(f"Error saving data to file: {e}")

    def load_config(self):
        with open(f"{script_directory}/config.json", "r") as json_file:
            self.config = json.load(json_file)

            print(f"Data successfully loaded from {json_file}")
            return self.config

    def get_slider_value(self, index):
        # Get a specific slider value by index
        if 0 <= index < len(self.slider_vars):
            return self.slider_vars[index].get()
        return None

    def on_closing(self):
        # Clean shutdown
        self.running = False
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=1.0)
        self.root.destroy()

    def run(self):
        # Start the GUI
        self.root.mainloop()

# Example usage
if __name__ == "__main__":
    # Create the GUI
    gui = RealTimeGUI()

    # Optional: Add more sliders before running
    # gui.add_slider("Custom Slider", min_val=0, max_val=1000, default_val=500)

    # Start the GUI
    gui.run()
