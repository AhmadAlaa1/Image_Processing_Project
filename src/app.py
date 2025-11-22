import tkinter as tk

from ui.controls import ImageApp



def main():
    root = tk.Tk()
    root.title("Image Processing Studio")
    root.geometry("1100x720")
    root.configure(bg="#111827")
    ImageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
