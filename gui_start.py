from mtslinkdownloader.gui import main
from mtslinkdownloader.utils import restore_terminal

if __name__ == "__main__":
    try:
        main()
    finally:
        restore_terminal()
