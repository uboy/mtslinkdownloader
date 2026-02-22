import PyInstaller.__main__
import os
import shutil
import platform
import customtkinter

# 1. Настройка параметров сборки
ctk_path = os.path.dirname(customtkinter.__file__)
app_name = 'mtslinkdownloader'
system_platform = platform.system().lower() # windows, linux, darwin
if system_platform == 'darwin':
    system_platform = 'macos'

print(f"--- Starting build for {system_platform} ---")

args = [
    'gui_start.py',
    f'--name={app_name}',
    '--onedir',
    '--windowed',
    f'--add-data={ctk_path}{os.pathsep}customtkinter',
    '--clean',
]

# Добавляем метаданные только для Windows
if system_platform == 'windows':
    if os.path.exists('file_version_info.txt'):
        args.append('--version-file=file_version_info.txt')

# 2. Запуск PyInstaller
PyInstaller.__main__.run(args)

# 3. Создание ZIP-архива
print(f"--- Packaging into ZIP ---")
dist_path = os.path.join('dist', app_name)
archive_name = f"{app_name}-{system_platform}" # например mtslinkdownloader-windows
archive_output_path = os.path.join('dist', archive_name)

# Удаляем старый архив если есть
if os.path.exists(archive_output_path + '.zip'):
    os.remove(archive_output_path + '.zip')

# Архивируем папку dist/mtslinkdownloader в dist/mtslinkdownloader-platform.zip
try:
    shutil.make_archive(archive_output_path, 'zip', 'dist', app_name)
    print(f"\nSUCCESS! Final archive created: {archive_output_path}.zip")
except Exception as e:
    print(f"\nError during archiving: {e}")
