import tkinter as tk
from emotion_app.ui import EmotionRecommenderApp

def main():
    root = tk.Tk()
    app = EmotionRecommenderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()