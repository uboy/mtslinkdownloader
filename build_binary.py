import PyInstaller.__main__
import os
import shutil
import platform
import customtkinter

# 1. Настройка параметров
ctk_path = os.path.dirname(customtkinter.__file__)
app_name = 'mtslinkdownloader'
system_platform = platform.system().lower()
if system_platform == 'darwin': system_platform = 'macos'

is_windows = system_platform == 'windows'
exe_ext = '.exe' if is_windows else ''

print(f"--- Starting build for {system_platform} ---")

# Общие аргументы
common_args = [
    '--onedir',
    '--clean',
    f'--add-data={ctk_path}{os.pathsep}customtkinter',
]
if is_windows and os.path.exists('file_version_info.txt'):
    common_args.append('--version-file=file_version_info.txt')

# 2. Сборка CLI версии
print("\n>>> Building CLI version...")
cli_name = f"{app_name}-cli"
PyInstaller.__main__.run([
    'cli_start.py',
    f'--name={cli_name}',
    '--console', # Консольное окно нужно для CLI
] + common_args)

# 3. Сборка GUI версии
print("\n>>> Building GUI version...")
PyInstaller.__main__.run([
    'gui_start.py',
    f'--name={app_name}',
    '--windowed', # Скрываем консоль для GUI
] + common_args)

# 4. Перенос CLI в папку GUI
print("\n>>> Merging binaries...")
src_cli_exe = os.path.join('dist', cli_name, f"{cli_name}{exe_ext}")
dst_folder = os.path.join('dist', app_name)
dst_cli_exe = os.path.join(dst_folder, f"{cli_name}{exe_ext}")

if os.path.exists(src_cli_exe):
    shutil.copy2(src_cli_exe, dst_cli_exe)
    print(f"Copied {cli_name} to {app_name} folder.")

# 5. Создание ZIP-архива
print(f"\n--- Packaging into ZIP ---")
archive_name = f"{app_name}-{system_platform}"
archive_output_path = os.path.join('dist', archive_name)

if os.path.exists(archive_output_path + '.zip'):
    os.remove(archive_output_path + '.zip')

try:
    shutil.make_archive(archive_output_path, 'zip', 'dist', app_name)
    print(f"\nSUCCESS! Final archive created: {archive_output_path}.zip")
    print(f"Inside you will find:")
    print(f"  - {app_name}{exe_ext} (Graphical Interface)")
    print(f"  - {cli_name}{exe_ext} (Command Line Interface)")
except Exception as e:
    print(f"\nError during archiving: {e}")
