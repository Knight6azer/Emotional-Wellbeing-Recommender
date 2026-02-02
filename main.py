import customtkinter as ctk
from emotion_app.ui import EmotionRecommenderApp

def main():
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("dark-blue")
    
    root = ctk.CTk()
    app = EmotionRecommenderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()