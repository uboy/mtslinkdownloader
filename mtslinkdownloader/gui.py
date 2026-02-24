import customtkinter as ctk
import threading
import logging
import sys
import os
import re
import time
from datetime import datetime
from typing import Optional

# Pre-import to avoid issues
from mtslinkdownloader.webinar import fetch_webinar_data
from mtslinkdownloader.utils import (
    create_directory_if_not_exists,
    get_base_path,
    initialize_logger,
    restore_terminal,
)
from mtslinkdownloader.cli import extract_ids_from_url
from mtslinkdownloader.processor import process_composite_video, request_stop
from mtslinkdownloader.downloader import construct_json_data_url, fetch_json_data

class TextRedirector:
    """Redirects stdout/stderr to a CTkTextbox with line-replacement support for tqdm."""
    def __init__(self, textbox):
        self.textbox = textbox

    def isatty(self):
        """Pretend to be a terminal for tqdm/logging compatibility."""
        return False

    def write(self, s):
        if not s: return
        self.textbox.after(0, self._safe_write, s)

    def _safe_write(self, s):
        try:
            self.textbox.configure(state="normal")
            if '\r' in s:
                parts = s.split('\r')
                if parts[-1].strip():
                    self.textbox.delete("end-2l", "end-1l")
                    self.textbox.insert("end", parts[-1] + "\n")
            else:
                self.textbox.insert("end", s)
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
        self.geometry("1000x850")
        ctk.set_appearance_mode("dark")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(8, weight=1)

        # UI Elements
        self.label = ctk.CTkLabel(self, text="MTS Link Webinar Downloader", font=ctk.CTkFont(size=24, weight="bold"))
        self.label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.url_label = ctk.CTkLabel(self, text="Webinar URL:", font=ctk.CTkFont(weight="bold"))
        self.url_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.url_entry = ctk.CTkEntry(self, placeholder_text="https://my.mts-link.ru/...", width=960)
        self.url_entry.grid(row=2, column=0, padx=20, pady=(0, 10))

        self.settings_frame = ctk.CTkFrame(self)
        self.settings_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.settings_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        self.sid_label = ctk.CTkLabel(self.settings_frame, text="Session ID (Optional):")
        self.sid_label.grid(row=0, column=0, padx=10, pady=(5, 0), sticky="w")
        self.session_id = ctk.CTkEntry(self.settings_frame)
        self.session_id.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.st_label = ctk.CTkLabel(self.settings_frame, text="Start offset (seconds):")
        self.st_label.grid(row=0, column=1, padx=10, pady=(5, 0), sticky="w")
        self.start_time = ctk.CTkEntry(self.settings_frame)
        self.start_time.insert(0, "0")
        self.start_time.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="ew")

        self.dur_label = ctk.CTkLabel(self.settings_frame, text="Limit duration (seconds):")
        self.dur_label.grid(row=0, column=2, padx=10, pady=(5, 0), sticky="w")
        self.max_duration = ctk.CTkEntry(self.settings_frame)
        self.max_duration.grid(row=1, column=2, padx=10, pady=(0, 10), sticky="ew")

        self.q_label = ctk.CTkLabel(self.settings_frame, text="Video Quality:")
        self.q_label.grid(row=0, column=3, padx=10, pady=(5, 0), sticky="w")
        self.quality_var = ctk.StringVar(value="FULL HD (1920x1080)")
        self.quality_menu = ctk.CTkOptionMenu(self.settings_frame, values=["FULL HD (1920x1080)", "HD (1280x720)"], variable=self.quality_var)
        self.quality_menu.grid(row=1, column=3, padx=10, pady=(0, 10), sticky="ew")

        self.log_label = ctk.CTkLabel(self.settings_frame, text="Log level:")
        self.log_label.grid(row=0, column=4, padx=10, pady=(5, 0), sticky="w")
        self.log_level_var = ctk.StringVar(value="INFO")
        self.log_level_menu = ctk.CTkOptionMenu(
            self.settings_frame,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            variable=self.log_level_var,
        )
        self.log_level_menu.grid(row=1, column=4, padx=10, pady=(0, 10), sticky="ew")

        self.hide_silent = ctk.CTkCheckBox(self, text="Aggressive optimization: Hide non-speaking participants")
        self.hide_silent.select() 
        self.hide_silent.grid(row=4, column=0, padx=20, pady=5)

        self.hint = ctk.CTkLabel(self, text="Tip: 0 in 'Start offset' means from the beginning. Duration is in seconds (600 = 10 min).", 
                                font=ctk.CTkFont(size=11, slant="italic"), text_color="gray")
        self.hint.grid(row=5, column=0, padx=20, pady=0)

        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.grid(row=6, column=0, padx=20, pady=15, sticky="ew")
        self.btn_frame.grid_columnconfigure((0, 1), weight=1)

        self.download_btn = ctk.CTkButton(self.btn_frame, text="START DOWNLOAD & RENDER", command=self.start_task, 
                                         font=ctk.CTkFont(size=16, weight="bold"), height=50, fg_color="#2c8c2c", hover_color="#1e5e1e")
        self.download_btn.grid(row=0, column=0, padx=10, pady=0, sticky="ew")

        self.stop_btn = ctk.CTkButton(self.btn_frame, text="STOP", command=self.stop_task, 
                                      font=ctk.CTkFont(size=16, weight="bold"), height=50, fg_color="#a83232", hover_color="#7a2424", state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=10, pady=0, sticky="ew")

        self.console_label = ctk.CTkLabel(self, text="Progress & Logs:", font=ctk.CTkFont(weight="bold"))
        self.console_label.grid(row=7, column=0, padx=20, pady=(5, 0), sticky="w")
        
        self.console = ctk.CTkTextbox(self, state="disabled", font=ctk.CTkFont(family="Consolas", size=12))
        self.console.grid(row=8, column=0, padx=20, pady=(0, 20), sticky="nsew")

        # REDIRECTION
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self.redirector = TextRedirector(self.console)
        sys.stdout = self.redirector
        sys.stderr = self.redirector

        initialize_logger(force=True, stream=sys.stderr, level=self.log_level_var.get())
        logging.getLogger("httpx").setLevel(logging.WARNING)

        print(f"App initialized. Base path: {get_base_path()}")
        print("Ready.\n")

    def destroy(self):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        super().destroy()

    def stop_task(self):
        self.stop_btn.configure(state="disabled", text="Stopping...")
        request_stop()

    def start_task(self):
        url = self.url_entry.get().strip()
        if not url: return
        self.download_btn.configure(state="disabled", text="PROCESSING...")
        self.stop_btn.configure(state="normal")
        
        s_id = self.session_id.get().strip() or None
        try:
            t_start = float(self.start_time.get() or 0)
            t_max = float(self.max_duration.get()) if self.max_duration.get().strip() else None
        except ValueError:
            print("Error: Invalid numeric values.")
            self.download_btn.configure(state="normal", text="START")
            self.stop_btn.configure(state="disabled")
            return
            
        quality = "1080p" if "1920" in self.quality_var.get() else "720p"
        initialize_logger(force=True, stream=sys.stderr, level=self.log_level_var.get())
        thread = threading.Thread(target=self.run_process, args=(url, s_id, t_start, t_max, self.hide_silent.get(), quality), daemon=True)
        thread.start()

    def run_process(self, url, s_id, t_start, t_max, h_silent, quality):
        run_started_ts = time.perf_counter()
        run_started_at = datetime.now()
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
            print(f"  Started at:   {run_started_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Start offset: {t_start} seconds")
            print(f"  Max duration: {t_max if t_max else 'Full record'} seconds")
            print(f"  Quality:      {quality.upper()}")
            print(f"  Log level:    {self.log_level_var.get()}")
            print(f"  Session ID:   {'Set' if s_id else 'Not set'}")
            print(f"  Optimization: {'Aggressive (Hide Silent)' if h_silent else 'Standard'}")
            print("-" * 40)
            
            process_composite_video(directory, json_data, output_video_path, t_max, hide_silent=h_silent, start_time=t_start, quality=quality)
            
            full_path = os.path.abspath(output_video_path)
            run_finished_at = datetime.now()
            elapsed = int(time.perf_counter() - run_started_ts)
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            print(f"\n!!! SUCCESS !!! Video is ready.")
            print(f"Path: {full_path}")
            print(f"Finished at: {run_finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total elapsed: {h:02d}:{m:02d}:{s:02d}")
        except Exception as e:
            if "interrupted" in str(e).lower():
                run_finished_at = datetime.now()
                elapsed = int(time.perf_counter() - run_started_ts)
                h, rem = divmod(elapsed, 3600)
                m, s = divmod(rem, 60)
                print("\n[!] Process stopped by user.")
                print(f"Stopped at: {run_finished_at.strftime('%Y-%m-%d %H:%M:%S')} | Elapsed: {h:02d}:{m:02d}:{s:02d}")
            else:
                run_finished_at = datetime.now()
                elapsed = int(time.perf_counter() - run_started_ts)
                h, rem = divmod(elapsed, 3600)
                m, s = divmod(rem, 60)
                print(f"\nFATAL ERROR: {e}")
                print(f"Failed at: {run_finished_at.strftime('%Y-%m-%d %H:%M:%S')} | Elapsed: {h:02d}:{m:02d}:{s:02d}")
        finally:
            self.download_btn.after(0, self._reset_buttons)

    def _reset_buttons(self):
        self.download_btn.configure(state="normal", text="START DOWNLOAD & RENDER")
        self.stop_btn.configure(state="disabled", text="STOP")

def main():
    try:
        app = App()
        app.mainloop()
    finally:
        restore_terminal()

if __name__ == "__main__":
    main()
