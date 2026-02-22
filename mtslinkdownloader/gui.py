import customtkinter as ctk
import threading
import logging
import sys
import os
import re
from typing import Optional

# Pre-import to avoid issues
from mtslinkdownloader.webinar import fetch_webinar_data
from mtslinkdownloader.utils import get_base_path, create_directory_if_not_exists
from mtslinkdownloader.cli import extract_ids_from_url
from mtslinkdownloader.processor import process_composite_video
from mtslinkdownloader.downloader import construct_json_data_url, fetch_json_data

class TextRedirector:
    """Redirects stdout/stderr to a CTkTextbox with line-replacement support for tqdm."""
    def __init__(self, textbox):
        self.textbox = textbox

    def write(self, s):
        if not s: return
        # Use after() to ensure UI updates happen in the main thread
        self.textbox.after(0, self._safe_write, s)

    def _safe_write(self, s):
        try:
            self.textbox.configure(state="normal")
            if '\r' in s:
                # Handle tqdm carriage return by overwriting the last line
                # We split by \r and take the last part
                parts = s.split('\r')
                if parts[-1].strip():
                    # Delete the last line before inserting the new progress
                    self.textbox.delete("end-2l", "end-1l")
                    self.textbox.insert("end", parts[-1] + "\n")
            else:
                self.textbox.insert("end", s)
            
            # Limit scrollback to 5000 lines to prevent memory issues
            if int(self.textbox.index('end-1c').split('.')[0]) > 5000:
                self.textbox.delete('1.0', '2.0')
                
            self.textbox.see("end")
            self.textbox.configure(state="disabled")
        except Exception: pass

    def flush(self):
        pass

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("MTS Link Downloader")
        self.geometry("1000x800")
        ctk.set_appearance_mode("dark")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(7, weight=1)

        # UI Elements
        self.label = ctk.CTkLabel(self, text="MTS Link Webinar Downloader", font=ctk.CTkFont(size=24, weight="bold"))
        self.label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.url_label = ctk.CTkLabel(self, text="Webinar URL:", font=ctk.CTkFont(weight="bold"))
        self.url_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.url_entry = ctk.CTkEntry(self, placeholder_text="https://my.mts-link.ru/...", width=960)
        self.url_entry.grid(row=2, column=0, padx=20, pady=(0, 10))

        self.settings_frame = ctk.CTkFrame(self)
        self.settings_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.settings_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.sid_label = ctk.CTkLabel(self.settings_frame, text="Session ID (Optional):")
        self.sid_label.grid(row=0, column=0, padx=10, pady=(5, 0), sticky="w")
        self.session_id = ctk.CTkEntry(self.settings_frame)
        self.session_id.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.st_label = ctk.CTkLabel(self.settings_frame, text="Start offset (seconds):")
        self.st_label.grid(row=0, column=1, padx=10, pady=(5, 0), sticky="w")
        self.start_time = ctk.CTkEntry(self.settings_frame)
        self.start_time.insert(0, "0")
        self.start_time.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="ew")

        self.dur_label = ctk.CTkLabel(self.settings_frame, text="Max duration (seconds):")
        self.dur_label.grid(row=0, column=2, padx=10, pady=(5, 0), sticky="w")
        self.max_duration = ctk.CTkEntry(self.settings_frame)
        self.max_duration.grid(row=1, column=2, padx=10, pady=(0, 10), sticky="ew")

        self.hide_silent = ctk.CTkCheckBox(self, text="Aggressive optimization: Hide non-speaking participants")
        self.hide_silent.select() 
        self.hide_silent.grid(row=4, column=0, padx=20, pady=5)

        self.download_btn = ctk.CTkButton(self, text="START DOWNLOAD & RENDER", command=self.start_task, 
                                         font=ctk.CTkFont(size=16, weight="bold"), height=50, fg_color="#2c8c2c", hover_color="#1e5e1e")
        self.download_btn.grid(row=5, column=0, padx=20, pady=15)

        self.console_label = ctk.CTkLabel(self, text="Progress & Logs:", font=ctk.CTkFont(weight="bold"))
        self.console_label.grid(row=6, column=0, padx=20, pady=(5, 0), sticky="w")
        
        self.console = ctk.CTkTextbox(self, state="disabled", font=ctk.CTkFont(family="Consolas", size=12))
        self.console.grid(row=7, column=0, padx=20, pady=(0, 20), sticky="nsew")

        # REDIRECTION MAGIC
        self.redirector = TextRedirector(self.console)
        sys.stdout = self.redirector
        sys.stderr = self.redirector

        # Initial logging setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%H:%M:%S', stream=sys.stderr)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        print(f"App initialized. Base path: {get_base_path()}")
        print("Ready. Paste your link and click Start.\n")

    def start_task(self):
        url = self.url_entry.get().strip()
        if not url: return
        self.download_btn.configure(state="disabled", text="PROCESSING...")
        
        s_id = self.session_id.get().strip() or None
        try:
            t_start = float(self.start_time.get() or 0)
            t_max = float(self.max_duration.get()) if self.max_duration.get().strip() else None
        except ValueError:
            print("Error: Invalid numeric values.")
            self.download_btn.configure(state="normal", text="START")
            return
            
        thread = threading.Thread(target=self.run_process, args=(url, s_id, t_start, t_max, self.hide_silent.get()), daemon=True)
        thread.start()

    def run_process(self, url, s_id, t_start, t_max, h_silent):
        try:
            event_sessions, record_id = extract_ids_from_url(url)
            if not event_sessions:
                print("INVALID URL format.")
                return

            json_url = construct_json_data_url(event_sessions, record_id)
            print("Fetching metadata...")
            json_data = fetch_json_data(json_url, s_id)
            if not json_data: return

            name = re.sub(r'[\s\/:*?"<>|]+', '_', json_data['name'])
            directory = create_directory_if_not_exists(name)
            output_video_path = os.path.join(directory, f'{name}.mp4')
            
            print(f"Working on: {json_data['name']}")
            print("-" * 40)
            print(f"  Start offset: {t_start} seconds")
            print(f"  Max duration: {t_max if t_max else 'Full record'} seconds")
            print(f"  Session ID:   {'Set' if s_id else 'Not set'}")
            print(f"  Optimization: {'Aggressive (Hide Silent)' if h_silent else 'Standard'}")
            print("-" * 40)
            
            process_composite_video(directory, json_data, output_video_path, t_max, hide_silent=h_silent, start_time=t_start)
            print("\n!!! SUCCESS !!! Video is ready.")
        except Exception as e:
            print(f"\nFATAL ERROR: {e}")
        finally:
            self.download_btn.after(0, lambda: self.download_btn.configure(state="normal", text="START DOWNLOAD & RENDER"))

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
