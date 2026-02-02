import tkinter as tk
import customtkinter as ctk
from tkinter import ttk, scrolledtext, messagebox
import cv2
import time
import threading
import random
import zipfile
import os
import tempfile
import numpy as np
import datetime

# Disable TensorFlow warnings and oneDNN to prevent crashes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

try:
    import matplotlib
    matplotlib.use('TkAgg')  # Set backend early to avoid issues
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except Exception as e:
    print(f"Warning: Matplotlib not available: {e}")
    plt = None
    FigureCanvasTkAgg = None
    MATPLOTLIB_AVAILABLE = False
from collections import Counter
from PIL import Image, ImageTk
import wave
import queue
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None
    PYAUDIO_AVAILABLE = False
    print("Warning: pyaudio not available. Audio features will be disabled.")

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    sr = None
    SPEECH_AVAILABLE = False
    print("Warning: speech_recognition not available. Voice input will be disabled.")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not available. Text-to-speech will be disabled.")

# Import real emotion models - DISABLED to prevent crashes
EMOTION_MODELS_AVAILABLE = False
load_facial_model = None
load_voice_model = None
FacialEmotionModel = None
VoiceEmotionModel = None

# Temporarily disable real models to prevent TensorFlow crashes
print("[SAFETY] Real emotion models disabled to prevent system crashes")
print("[SAFETY] Using safe fallback methods only")

# try:
#     from emotion_models import load_facial_model, load_voice_model, FacialEmotionModel, VoiceEmotionModel
#     EMOTION_MODELS_AVAILABLE = True
# except ImportError:
#     # Emotion models not available - will use fallback methods
#     pass

# --- Constants and Configuration ---
APP_TITLE = "Emotionally Aware Wellbeing Recommender"
WINDOW_SIZE = "1200x800"
BACKGROUND_COLOR = "#2c3e50"
TEXT_COLOR = "#ecf0f1"
FONT_FAMILY = "Inter"
EMOTIONS = ["sad", "angry", "disgust", "fear", "happy", "neutral", "surprise"]

# Import Modules
from .vision import VisionAnalyzer
from .text_analysis import TextAnalyzer
from .audio_analysis import AudioAnalyzer
from .fusion import EmotionFusion
from .recommendations import RecommendationEngine

# --- Main Application Class ---

# Note: This function is no longer used in the main flow, but kept for compatibility


# --- Main Application Class ---
class EmotionRecommenderApp:
    def __init__(self, root):
        try:
            self.root = root
            self.root.title(APP_TITLE)
            self.root.geometry(WINDOW_SIZE)
            # self.root.configure(bg=BACKGROUND_COLOR) # Managed by CustomTkinter
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

            # --- Initialize Modular Components ---
            print("Initializing emotion detection modules...")
            
            # 1. Vision Module
            self.vision_analyzer = VisionAnalyzer()
            
            # 2. Text Module
            self.text_analyzer = TextAnalyzer()
            
            # 3. Audio Module
            self.audio_analyzer = AudioAnalyzer()
            
            # 4. Fusion Module
            self.fusion = EmotionFusion(alpha=0.7) # Alpha for smoothing
            
            # 5. Recommendation Engine
            self.recommender = RecommendationEngine()
            
            print("[OK] All modules initialized with fallback support")
            
            # --- State Variables ---
            self.is_camera_running = False
            self.video_thread = None
            self.audio_thread = None
            self.transcription_thread = None
            self.face_detected = False
            self.face_dialog_shown = False
            self.is_transcribing = False
            self.is_audio_running = False
            self.is_audio_visualization_running = True
            self.current_frame = None
            self.audio_visualization_thread = None
            self.last_text_input = ""
            
            # Enhanced voice recording state
            self.is_voice_recording = False
            self.voice_recording_thread = None
            # Shared audio resources
            self.audio_queue = queue.Queue(maxsize=50)
            self.pyaudio_instance = None
            self.audio_stream = None
            self.audio_reader_thread = None
            self.is_audio_reader_running = False
            self.audio_chunk_size = 1024
            self.audio_rate = 44100
            self.audio_data = []
            self.recognition_engine = None
            
            # Initialize emotion containers
            self.visual_emotions = {e: 0.0 for e in EMOTIONS}
            self.text_emotions = {e: 0.0 for e in EMOTIONS}
            self.audio_emotions = {e: 0.0 for e in EMOTIONS}
            self.combined_emotions = {e: 0.0 for e in EMOTIONS} # Initial combined
            
            # Initialize emotion history tracking
            self.emotion_history = {e: [] for e in EMOTIONS}
            self.timestamps = []
            self.analysis_count = 0
            self.last_update_time = datetime.datetime.now()
            self.last_frame_analysis = datetime.datetime.now()
            self.last_analytics_update = datetime.datetime.now()

            # --- UI Setup ---
            try:
                print("Creating UI header...")
                self.create_header()
                print("Creating UI widgets...")
                self.create_widgets()
                print("UI created successfully")
            except Exception as e:
                print(f"Error creating UI: {e}")
                self.create_minimal_ui()
            
            # Create custom styles
            try:
                self.style = ttk.Style()
                try:
                    self.style.theme_use("clam")
                except Exception:
                    pass
                self.style.configure("Dominant.Horizontal.TProgressbar", 
                                    background="#e74c3c", 
                                    troughcolor="#34495e")
                self.style.configure("Horizontal.TProgressbar", thickness=12)
                self.style.configure("TButton", padding=6, font=(FONT_FAMILY, 10, "bold"))
            except Exception as e:
                print(f"Error configuring styles: {e}")
            
            # Initialize voice recognition
            try:
                if SPEECH_AVAILABLE and sr is not None:
                    self.recognition_engine = sr.Recognizer()
                    self.recognition_engine.energy_threshold = 300
                    self.recognition_engine.dynamic_energy_threshold = True
                else:
                    self.recognition_engine = None
            except Exception as e:
                print(f"Error initializing speech recognition: {e}")
                self.recognition_engine = None
            
            # Start processes
            try:
                print("Starting UI update process...")
                self.update_emotion_display()
                print("Initializing shared audio stream...")
                self.start_audio_processing()
                print("Starting audio analysis...")
                self.start_audio_analysis()
                print("Starting transcription...")
                self.start_transcription_if_available()
                print("Starting audio visualization...")
                self.start_audio_visualization()
            except Exception as e:
                print(f"Error starting processes: {e}")
            
            # Bind focus events
            try:
                self.root.bind("<FocusIn>", self.on_focus_in)
                self.root.bind("<FocusOut>", self.on_focus_out)
            except Exception:
                pass
            self.update_timer = None
            
        except Exception as e:
            print(f"Critical error during initialization: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Initialization Error", f"Error: {str(e)}")

        
    def on_focus_in(self, event=None):
        """Handle window focus in event - restore normal update frequency"""
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(100, self.update_emotion_display)
        
    def on_focus_out(self, event=None):
        """Handle window focus out event - reduce update frequency to save resources"""
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(500, self.update_emotion_display)
    
    def create_minimal_ui(self):
        """Create minimal UI as fallback when main UI creation fails"""
        try:
            print("Creating minimal UI fallback...")
            
            # Create basic frame
            main_frame = tk.Frame(self.root, bg=BACKGROUND_COLOR)
            main_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Add basic title
            title_label = tk.Label(main_frame, text="Emotion Recommender", 
                                    bg=BACKGROUND_COLOR, fg=TEXT_COLOR, 
                                    font=(FONT_FAMILY, 16, "bold"))
            title_label.pack(pady=10)
            
            # Add status message
            status_label = tk.Label(main_frame, 
                                    text="Running in minimal mode with safe fallbacks",
                                    bg=BACKGROUND_COLOR, fg="#f39c12",
                                    font=(FONT_FAMILY, 10))
            status_label.pack(pady=5)
            
            # Add basic emotion display
            emotion_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
            emotion_frame.pack(fill="x", pady=10)
            
            self.dominant_emotion_label = tk.Label(emotion_frame, 
                                                    text="Ready - Using safe detection",
                                                    bg=BACKGROUND_COLOR, fg="#3498db",
                                                    font=(FONT_FAMILY, 12))
            self.dominant_emotion_label.pack()
            
            # Add text input area
            text_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
            text_frame.pack(fill="both", expand=True, pady=10)
            
            text_label = tk.Label(text_frame, text="How are you feeling?",
                                    bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
                                    font=(FONT_FAMILY, 10))
            text_label.pack(anchor="w", pady=2)
            
            self.text_input = tk.Text(text_frame, height=4, width=40,
                                    font=(FONT_FAMILY, 10))
            self.text_input.pack(fill="x")
            
            # Add analyze button
            analyze_button = ttk.Button(main_frame, text="Analyze Emotion",
                                        command=self.analyze_text_emotion)
            analyze_button.pack(pady=5)
            
            # Add recommendation area
            rec_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
            rec_frame.pack(fill="both", expand=True, pady=10)
            
            rec_label = tk.Label(rec_frame, text="Recommendations:",
                                    bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
                                    font=(FONT_FAMILY, 10, "bold"))
            rec_label.pack(anchor="w", pady=2)
            
            self.recommendation_text = tk.Text(rec_frame, height=6, width=40,
                                                font=(FONT_FAMILY, 9))
            self.recommendation_text.pack(fill="both", expand=True)
            
            print("[OK] Minimal UI created successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to create minimal UI: {e}")
            raise
    
    def validate_system(self):
        """Validate system capabilities (non-blocking, no dialogs during init)"""
        warnings = []
        
        # Check for camera availability
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                warnings.append("Camera not available - visual emotion detection disabled")
            else:
                cap.release()
        except Exception:
            warnings.append("Camera not available - visual emotion detection disabled")
        
        # Check for audio availability
        if not PYAUDIO_AVAILABLE:
            warnings.append("Audio system not available - voice features disabled")
        else:
            try:
                p = pyaudio.PyAudio()
                p.terminate()
            except Exception:
                warnings.append("Audio system not available - voice features disabled")
        
        # Check for speech recognition
        if not SPEECH_AVAILABLE:
            warnings.append("Speech recognition not available - voice input disabled")
        
        # Check for text-to-speech
        if not TTS_AVAILABLE:
            warnings.append("Text-to-speech not available - voice output disabled")
        
        # Log warnings but don't show dialog during initialization
        # Dialog will be shown later if needed
        if warnings:
            print(f"System status: {len(warnings)} optional features unavailable")
            for w in warnings:
                print(f"  - {w}")
            print("  Application will work with available features\n")
        
        # Store warnings for later display (after GUI is ready)
        if not hasattr(self, 'system_warnings'):
            self.system_warnings = []
        self.system_warnings = warnings
        # Show warnings dialog after a delay (non-blocking)
        if warnings:
            try:
                self.root.after(2000, self._show_system_warnings)
            except Exception as e:
                print(f"Error scheduling warning dialog: {e}")
    
    def _show_system_warnings(self):
        """Show system warnings dialog (called after GUI is ready)"""
        try:
            if not hasattr(self, 'system_warnings') or not self.system_warnings:
                return
            
            warning_text = "System Status:\n\n" + "\n".join(f"- {w}" for w in self.system_warnings)
            warning_text += "\n\nThe application will work with available features."
            
            # Show warning dialog
            dialog = ctk.CTkToplevel(self.root)
            dialog.title("System Status")
            dialog.geometry("500x300")
            
            # Make dialog modal
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Add message
            message = ctk.CTkLabel(dialog, text=warning_text,
                                text_color="#f39c12", font=(FONT_FAMILY, 11),
                                justify="left", wraplength=450)
            message.pack(pady=20, padx=20, fill="both", expand=True)
            
            # Add OK button
            ok_button = ctk.CTkButton(dialog, text="Continue", command=dialog.destroy)
            ok_button.pack(pady=10)
            
            # Auto-close after 10 seconds
            self.root.after(10000, dialog.destroy)
        except Exception as e:
            print(f"Error showing system warnings: {e}")
    
    def _load_models_async(self):
        """Legacy model loading - deprecated"""
        pass

    def create_header(self):
        header = ctk.CTkFrame(self.root, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(12, 0))
        
        title = ctk.CTkLabel(header, text=APP_TITLE, font=(FONT_FAMILY, 24, "bold"))
        title.pack(side="left")
        
        subtitle = ctk.CTkLabel(header, text="Multimodal wellbeing insights", 
                               text_color="#95a5a6", font=(FONT_FAMILY, 14))
        subtitle.pack(side="left", padx=(15, 0), pady=(8, 0))

    def create_styled_frame(self, parent, title):
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="both", expand=True, pady=10)
        
        title_label = ctk.CTkLabel(container, text=title, font=(FONT_FAMILY, 14, "bold"))
        title_label.pack(anchor="w", padx=5, pady=(0, 5))
        
        frame = ctk.CTkFrame(container)
        frame.pack(fill="both", expand=True, ipadx=10, ipady=10)
        return frame

    def create_widgets(self):
        """Creates and arranges all UI widgets."""
        main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # --- Left Panel: Inputs ---
        left_panel = ctk.CTkFrame(main_frame, fg_color="transparent")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Video
        video_frame = self.create_styled_frame(left_panel, "Visual Input (Camera)")
        # For video label, we use a standard Label inside a CTkFrame because CTkLabel handling of images can be tricky with continuous updates
        # But let's try CTkLabel first. If it flickers or has issues, we can revert this specific widget.
        # Actually standard tk.Label is often better for raw high-fps image updates in some cases, but CTkLabel should work.
        # Let's use tk.Label for the video stream component specifically to be safe, but styled.
        video_inner = tk.Label(video_frame, bg="black") 
        video_inner.pack(pady=10, padx=10, expand=True, fill="both")
        self.video_label = video_inner
        
        # Camera control buttons frame
        camera_buttons_frame = ctk.CTkFrame(video_frame, fg_color="transparent")
        camera_buttons_frame.pack(pady=(0, 10), fill="x")
        
        self.camera_button = ctk.CTkButton(camera_buttons_frame, text="Start Camera", command=self.toggle_camera)
        self.camera_button.pack(side="left", padx=(10, 5), expand=True)
        
        self.take_photo_button = ctk.CTkButton(camera_buttons_frame, text="Capture", command=self.take_photo, state="disabled")
        self.take_photo_button.pack(side="right", padx=(5, 10), expand=True)
        
        # Visual Emotions Dashboard
        self.visual_dashboard = self.create_styled_frame(left_panel, "Facial Emotion Dashboard")
        self.facial_emotion_bars = {}
        for emotion in EMOTIONS:
            frame = ctk.CTkFrame(self.visual_dashboard, fg_color="transparent")
            frame.pack(fill="x", padx=10, pady=2)
            
            label = ctk.CTkLabel(frame, text=emotion.capitalize(), width=70, anchor="w")
            label.pack(side="left")
            
            progress = ctk.CTkProgressBar(frame)
            progress.set(0)
            progress.pack(side="left", fill="x", expand=True, padx=(10, 0))
            
            self.facial_emotion_bars[emotion] = progress

        # Text Input Section with Enhanced UI
        text_frame = self.create_styled_frame(left_panel, "Text Input & Voice Recognition")
        
        # Voice control buttons frame
        voice_controls_frame = ctk.CTkFrame(text_frame, fg_color="transparent")
        voice_controls_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        # Voice recording button
        self.voice_button = ctk.CTkButton(voice_controls_frame, text="[MIC] Start Voice Input", command=self.toggle_voice_recording)
        self.voice_button.pack(side="left", padx=(0, 10))
        
        # Voice status indicator
        self.voice_status_label = ctk.CTkLabel(voice_controls_frame, text="Voice: Ready", text_color="#2ecc71", font=(FONT_FAMILY, 12, "bold"))
        self.voice_status_label.pack(side="left", padx=(0, 10))
        
        # Clear text button
        clear_button = ctk.CTkButton(voice_controls_frame, text="Clear Text", command=self.clear_text_input, fg_color="#e74c3c", hover_color="#c0392b") # Red color handling
        clear_button.pack(side="right")
        
        # Enhanced text input area
        self.text_input = ctk.CTkTextbox(
            text_frame, 
            height=150,
            font=(FONT_FAMILY, 14),
            wrap="word"
        )
        self.text_input.pack(pady=10, padx=10, expand=True, fill="both")
        self.text_input.bind("<KeyRelease>", self.update_text_analysis)
        
        # Add placeholder text logic
        self.text_input.insert("1.0", "Type how you feel or use voice input to express your emotions...")
        # Note: CTkTextbox doesn't have direct fg config for placeholder, we manage it via logic
        self.text_input.bind("<FocusIn>", self.on_text_focus_in)
        self.text_input.bind("<FocusOut>", self.on_text_focus_out)
        
        # Text Emotions Dashboard
        self.text_dashboard = self.create_styled_frame(left_panel, "Text Emotion Dashboard")
        self.text_emotion_bars = {}
        for emotion in EMOTIONS:
            frame = ctk.CTkFrame(self.text_dashboard, fg_color="transparent")
            frame.pack(fill="x", padx=10, pady=2)
            
            label = ctk.CTkLabel(frame, text=emotion.capitalize(), width=70, anchor="w")
            label.pack(side="left")
            
            progress = ctk.CTkProgressBar(frame)
            progress.set(0)
            progress.pack(side="left", fill="x", expand=True, padx=(10, 0))
            
            self.text_emotion_bars[emotion] = progress
            
        # Emotional Stability Score
        stability_frame = self.create_styled_frame(left_panel, "Emotional Stability")
        self.stability_score_label = ctk.CTkLabel(stability_frame, text="Calculating...", text_color="#3498db", font=(FONT_FAMILY, 14, "bold"))
        self.stability_score_label.pack(pady=5)
        self.stability_progress = ctk.CTkProgressBar(stability_frame, width=200)
        self.stability_progress.set(0)
        self.stability_progress.pack(pady=5, fill="x", padx=10)

        # --- Right Panel: Analysis & Recommendations ---
        right_panel = ctk.CTkFrame(main_frame, fg_color="transparent")
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # Audio Emotions Dashboard with Voice Controls
        self.audio_dashboard = self.create_styled_frame(right_panel, "Voice Emotion Dashboard")
        
        # Voice recording controls
        voice_control_frame = ctk.CTkFrame(self.audio_dashboard, fg_color="transparent")
        voice_control_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Audio capture button
        self.audio_capture_button = ctk.CTkButton(voice_control_frame, text="[AUDIO] Capture Voice Emotion", command=self.capture_audio_emotion)
        self.audio_capture_button.pack(side="left", padx=(0, 10))
        
        # Audio level indicator
        self.audio_level_label = ctk.CTkLabel(voice_control_frame, text="Audio Level: --", text_color="#f39c12", font=(FONT_FAMILY, 12))
        self.audio_level_label.pack(side="left")
        
        # Audio visualization canvas
        # Canvas is tk specific, but can live inside CTk. 
        self.audio_canvas = tk.Canvas(self.audio_dashboard, height=60, bg="#2c3e50", highlightthickness=0)
        self.audio_canvas.pack(fill="x", padx=10, pady=(0, 10))
        
        # Audio emotion bars
        self.audio_emotion_bars = {}
        for emotion in EMOTIONS:
            frame = ctk.CTkFrame(self.audio_dashboard, fg_color="transparent")
            frame.pack(fill="x", padx=10, pady=2)
            
            label = ctk.CTkLabel(frame, text=emotion.capitalize(), width=70, anchor="w")
            label.pack(side="left")
            
            progress = ctk.CTkProgressBar(frame)
            progress.set(0)
            progress.pack(side="left", fill="x", expand=True, padx=(10, 0))
            
            self.audio_emotion_bars[emotion] = progress

        # Combined Emotion with Enhanced Display
        combined_frame = self.create_styled_frame(right_panel, "Dominant Emotion Analysis")
        
        # Main emotion display
        emotion_display_frame = ctk.CTkFrame(combined_frame, fg_color="transparent")
        emotion_display_frame.pack(fill="x", pady=20, padx=20)
        
        self.dominant_emotion_label = ctk.CTkLabel(
            emotion_display_frame, 
            text="Calculating...", 
            text_color="#3498db", 
            font=(FONT_FAMILY, 28, "bold"),
            wraplength=400,
            justify="center"
        )
        self.dominant_emotion_label.pack(pady=10)
        
        # Confidence indicator
        self.confidence_label = ctk.CTkLabel(
            emotion_display_frame, 
            text="", 
            text_color="#95a5a6", 
            font=(FONT_FAMILY, 14, "normal")
        )
        self.confidence_label.pack(pady=(0, 10))
        
        # Modality agreement display
        self.agreement_label = ctk.CTkLabel(
            combined_frame, 
            text="", 
            text_color="#bdc3c7", 
            font=(FONT_FAMILY, 12, "normal"),
            wraplength=450,
            justify="center"
        )
        self.agreement_label.pack(pady=(0, 10), padx=20)

        # Enhanced Recommendation Display
        reco_frame = self.create_styled_frame(right_panel, "Personalized Wellbeing Recommendation")
        self.recommendation_label = ctk.CTkLabel(
            reco_frame, 
            text="Enter input to get a personalized recommendation.", 
            font=(FONT_FAMILY, 13, "normal"),
            wraplength=500,
            justify="left",
            anchor="w"
        )
        self.recommendation_label.pack(pady=20, padx=15, fill="both", expand=True)
        
        # Analytics Dashboard
        analytics_frame = self.create_styled_frame(right_panel, "Multimodal Emotion Graph")
        
        # Matplotlib logic remains largely the same, but we need to ensure colors match theme
        try:
            if not MATPLOTLIB_AVAILABLE or plt is None:
                raise ImportError("Matplotlib not available")
            
            # Use dark background for plot
            self.fig, self.ax = plt.subplots(figsize=(5, 3))
            
            # Set to a dark color matching the theme
            plot_bg = "#2b2b2b" # approximation of CTk dark
            self.fig.patch.set_facecolor(plot_bg)
            self.ax.set_facecolor(plot_bg)
            self.ax.tick_params(colors="white")
            self.ax.spines['bottom'].set_color('white')
            self.ax.spines['top'].set_color('white') 
            self.ax.spines['left'].set_color('white')
            self.ax.spines['right'].set_color('white')
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            
            self.ax.set_title("Emotion History", color="white")
            self.ax.set_xlabel("Time", color="white")
            self.ax.set_ylabel("Intensity", color="white")
            
            # Create canvas for matplotlib figure
            self.canvas = FigureCanvasTkAgg(self.fig, master=analytics_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            self.matplotlib_available = True
        except Exception as e:
            print(f"Warning: Matplotlib initialization failed: {e}")
            error_label = ctk.CTkLabel(
                analytics_frame,
                text=f"Analytics unavailable:\n{str(e)[:50]}...",
                text_color="#e74c3c",
                font=(FONT_FAMILY, 10)
            )
            error_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            self.matplotlib_available = False
            self.fig = None
            self.ax = None
            self.canvas = None
        
        # Add refresh button
        refresh_btn = ctk.CTkButton(analytics_frame, text="Refresh Analytics", command=self.update_analytics)
        refresh_btn.pack(pady=10)

    def toggle_camera(self):
        """Toggles the camera on/off"""
        if self.is_camera_running:
            self.is_camera_running = False
            self.camera_button.configure(text="Start Camera")
            self.take_photo_button.configure(state="disabled")
            if self.video_thread:
                self.video_thread.join()
            self.video_label.configure(image='', bg='black') # tk.Label uses config/configure
            self.video_label.configure(bg="black")
        else:
            self.is_camera_running = True
            self.camera_button.configure(text="Stop Camera")
            self.take_photo_button.configure(state="normal")
            self.video_thread = threading.Thread(target=self.video_loop)
            self.video_thread.daemon = True
            self.video_thread.start()
            
    def compute_text_confidence(self, text, emotion_scores):
        """Estimate confidence (0-1) for text emotion based on cues and intensity."""
        if not text:
            return 0.0
        text = text.strip()
        length_factor = min(1.0, max(0.2, len(text) / 120.0))
        max_score = max(emotion_scores.values()) if emotion_scores else 0.0
        exclamations = text.count("!")
        questions = text.count("?")
        uppercase_words = sum(1 for w in text.split() if len(w) > 2 and w.isupper())
        emojis_positive = [":)", ":D", ":)", ":)", ":)", ":)", ":)", ":)"]
        emojis_negative = [":(", ":(", ":(", ":(", ":(", ":(", ":(", ":("]
        emoji_boost = 0.0
        emoji_boost += 0.1 * sum(text.count(e) for e in emojis_positive)
        emoji_boost += 0.1 * sum(text.count(e) for e in emojis_negative)
        punctuation_boost = min(0.3, 0.08 * exclamations + 0.04 * questions)
        uppercase_boost = min(0.2, 0.05 * uppercase_words)
        confidence = max_score * 0.6 + length_factor * 0.2 + punctuation_boost + uppercase_boost + emoji_boost
        return max(0.0, min(1.0, confidence))
            
    def take_photo(self):
        """Captures a still image, saves it, and analyzes it for emotions"""
        if not self.is_camera_running:
            return
            
        # Get the current frame from the video feed
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            # Create directory if it doesn't exist
            os.makedirs('captured_photos', exist_ok=True)
            
            # Save the image with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            photo_path = f"captured_photos/photo_{timestamp}.jpg"
            cv2.imwrite(photo_path, self.current_frame)
            
            # Analyze the saved image
            emotions, processed_frame, face_detected = analyze_face_emotions(self.current_frame, self.visual_model)
            
            if face_detected:
                # Store the emotions from the photo
                self.captured_visual_emotions = emotions.copy()
                
                # Update the UI to show captured emotions
                self.update_facial_emotion_display(use_captured=True)
                
                # Show the captured image in a popup
                self.show_captured_photo(processed_frame)
                
                # Show confirmation message
                self.show_capture_confirmation(f"Photo captured and analyzed successfully!")
            else:
                # Show error message if no face detected
                self.show_error_message("No face detected in the photo. Please try again.")
        else:
            self.show_error_message("No frame available. Please wait for the camera to initialize.")
            
    def video_loop(self):
        """Main video processing loop using VisionAnalyzer"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Cannot open camera.")
                self.is_camera_running = False
                self.root.after(0, lambda: self.camera_button.config(text="Start Camera"))
                self.root.after(0, lambda: messagebox.showerror("Camera Error", 
                    "Could not access camera.\n\nPlease check:\n- Camera is connected\n- No other apps are using the camera\n- Camera permissions are granted"))
                return
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.is_camera_running = False
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Check if Vision Analyzer is available
        if not self.vision_analyzer:
             print("[WARN] Vision Analyzer not initialized")
             self.is_camera_running = False
             return

        last_analysis_time = 0
        analysis_interval = 3.0  # Seconds between analysis

        while self.is_camera_running:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("[DEBUG] Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Store frame for capture
                self.current_frame = frame.copy()
                
                # Analyze Frame periodically
                current_time = time.time()
                is_analyzing = (current_time - last_analysis_time >= analysis_interval)
                
                img_to_show = frame # Default to raw frame
                
                if is_analyzing:
                    last_analysis_time = current_time
                    # This returns (emotions, processed_frame, face_detected)
                    self.visual_emotions, processed_frame, face_detected = self.vision_analyzer.analyze_frame(frame)
                    
                    self.face_detected = face_detected
                    
                    # Use processed frame if available (shows boxes)
                    if processed_frame is not None:
                        img_to_show = processed_frame
                    
                    # Store last processed frame for "stickiness" if we wanted, 
                    # but for now we just show raw frame in between to remain smooth.
                    # Or we could just not update the video label with boxes in between.
                    
                    # Handle Face Not Detected UI only on analysis check
                    if not face_detected:
                         if not self.face_dialog_shown:
                              self.root.after(0, self.show_face_not_detected_dialog)
                              self.face_dialog_shown = True
                    else:
                         self.face_dialog_shown = False
                
                # If NOT analyzing, we can optionally overlay the LAST known emotion/box 
                # if we stored it, but simply showing the raw live feed is capable enough 
                # and prevents "stuck" boxes on moving faces. 
                
                # Convert for Tkinter
                try:
                    # Resize for display if needed (optional)
                    # frame = cv2.resize(frame, (640, 480)) 
                    
                    img = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img_tk = ImageTk.PhotoImage(image=img)
                except Exception as img_err:
                    print(f"[DEBUG] Image conversion error: {img_err}")
                    continue
                
                # Update UI on main thread
                def update_video():
                    try:
                        if hasattr(self, 'video_label') and hasattr(self, 'root'):
                            self.video_label.configure(image=img_tk) # Use configure
                            self.video_label.img = img_tk
                    except Exception as ui_err:
                        print(f"[DEBUG] UI update error: {ui_err}")
                
                self.root.after(0, update_video)
                
                # Performance delay
                time.sleep(0.01)

            except Exception as e:
                print(f"Error in video_loop: {e}")
                time.sleep(1)

        cap.release()
        self.visual_emotions = {e: 0.0 for e in EMOTIONS}
        
    def show_captured_photo(self, image):
        """Displays the captured photo in a popup window"""
        # Convert the image from BGR to RGB for tkinter
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create a popup window
        popup = ctk.CTkToplevel(self.root)
        popup.title("Captured Photo")
        popup.geometry("700x600")
        
        # Create a frame for the image
        frame = ctk.CTkFrame(popup, fg_color="transparent")
        frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Convert the image to a format tkinter can use
        img = Image.fromarray(image_rgb)
        img = ImageTk.PhotoImage(image=img)
        
        # Keep a reference to prevent garbage collection
        popup.img = img
        
        # Display the image - use tk.Label for image inside CTk
        label = tk.Label(frame, image=img, bg="#2b2b2b")
        label.pack(padx=10, pady=10)
        
        # Add emotion analysis results
        emotions_frame = ctk.CTkFrame(popup, fg_color="transparent")
        emotions_frame.pack(padx=20, pady=(0, 20), fill="x")
        
        # Display emotion scores
        for emotion, score in self.captured_visual_emotions.items():
            emotion_label = ctk.CTkLabel(emotions_frame, 
                                     text=f"{emotion.capitalize()}: {score:.2f}", 
                                     font=(FONT_FAMILY, 12))
            emotion_label.pack(side="left", padx=10)
        
        # Add a close button
        close_button = ctk.CTkButton(popup, text="Close", command=popup.destroy)
        close_button.pack(pady=(0, 20))

    def update_facial_emotion_display(self, use_captured=False):
        """Updates the facial emotion dashboard with visual feedback for dominant emotion."""
        # This function is now only called from the main thread
        try:
            # Find dominant facial emotion
            emotions_to_display = self.captured_visual_emotions if use_captured and hasattr(self, 'captured_visual_emotions') else self.visual_emotions
            
            if any(v > 0 for v in emotions_to_display.values()):
                dominant_emotion = max(emotions_to_display, key=emotions_to_display.get)
                
                # Update progress bars with visual feedback for dominant emotion
                for emotion, value in emotions_to_display.items():
                    # Set progress bar value (0.0 to 1.0)
                    self.facial_emotion_bars[emotion].set(value)
                    
                    # Highlight only the dominant emotion
                    if emotion == dominant_emotion:
                        self.facial_emotion_bars[emotion].configure(progress_color=self.get_emotion_color(emotion))
                    else:
                        self.facial_emotion_bars[emotion].configure(progress_color="#3498db") # Default blue
            else:
                # Reset all progress bars if no emotions detected
                for emotion in EMOTIONS:
                    self.facial_emotion_bars[emotion].set(0)
                    self.facial_emotion_bars[emotion].configure(progress_color="#3498db")
        except Exception as e:
            print(f"Error updating facial display: {e}")
    
    def show_face_not_detected_dialog(self):
        """Show a dialog when face is not detected in the frame"""
        # This function is now only called from the main thread
        try:
            dialog = ctk.CTkToplevel(self.root)
            dialog.title("Face Not Detected")
            dialog.geometry("300x150")
            
            # Make dialog modal
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Add message
            message = ctk.CTkLabel(dialog, text="No face detected in the frame.\nPlease position yourself in front of the camera.",
                                font=(FONT_FAMILY, 12))
            message.pack(pady=20, padx=20)
            
            # Add OK button
            ok_button = ctk.CTkButton(dialog, text="OK", command=dialog.destroy)
            ok_button.pack(pady=10)
            
            # Auto-close after 3 seconds
            self.root.after(3000, dialog.destroy)
        except Exception as e:
            print(f"Error showing face detect dialog: {e}")
        
    def show_capture_confirmation(self, message):
        """Show a confirmation message when emotion is captured"""
        # This function is called from the main thread (take_photo)
        try:
            dialog = ctk.CTkToplevel(self.root)
            dialog.title("Emotion Captured")
            dialog.geometry("300x120")
            
            # Make dialog modal
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Add message
            msg_label = ctk.CTkLabel(dialog, text=message,
                                text_color="#2ecc71", font=(FONT_FAMILY, 12, "bold"))
            msg_label.pack(pady=20, padx=20)
            
            # Auto-close after 2 seconds
            self.root.after(2000, dialog.destroy)
        except Exception as e:
            print(f"Error showing capture dialog: {e}")

    def show_error_message(self, message):
        """Shows an error message dialog"""
        # This function is called from the main thread (take_photo)
        try:
            dialog = ctk.CTkToplevel(self.root)
            dialog.title("Error")
            dialog.geometry("300x150")
            
            # Make dialog modal
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Add message
            msg_label = ctk.CTkLabel(dialog, text=message,
                                text_color="#e74c3c", font=(FONT_FAMILY, 12, "bold"))
            msg_label.pack(pady=20, padx=20)
            
            # Add OK button
            ok_button = ctk.CTkButton(dialog, text="OK", command=dialog.destroy)
            ok_button.pack(pady=10)
        except Exception as e:
            print(f"Error showing error dialog: {e}")
                
    def on_text_focus_in(self, event):
        """Handle text input focus in - remove placeholder text"""
        if self.text_input.get("1.0", "end-1c").strip() == "Type how you feel or use voice input to express your emotions...":
            self.text_input.delete("1.0", "end")
            self.text_input.configure(text_color=TEXT_COLOR)
    
    def on_text_focus_out(self, event):
        """Handle text input focus out - add placeholder if empty"""
        if not self.text_input.get("1.0", "end-1c").strip():
            self.text_input.insert("1.0", "Type how you feel or use voice input to express your emotions...")
            self.text_input.configure(text_color="#95a5a6")
    
    def clear_text_input(self):
        """Clear the text input area"""
        self.text_input.delete("1.0", "end")
        self.text_input.insert("1.0", "Type how you feel or use voice input to express your emotions...")
        self.text_input.configure(text_color="#95a5a6")
        self.update_text_analysis()
    
    def toggle_voice_recording(self):
        """Toggle voice recording on/off"""
        if not self.is_voice_recording:
            self.start_voice_recording()
        else:
            self.stop_voice_recording()
    
    def start_voice_recording(self):
        """Start voice recording for speech-to-text"""
        if not SPEECH_AVAILABLE or sr is None:
            messagebox.showerror("Error", "Speech recognition not available. Please install speech_recognition library.")
            return
        
        try:
            # Test microphone access first
            with sr.Microphone() as source:
                pass  # Just test if we can access the microphone
            
            self.is_voice_recording = True
            self.voice_button.configure(text="[STOP] Stop Voice Input")
            self.voice_status_label.configure(text="Voice: Starting...", text_color="#f39c12")
            
            # Start recording in a separate thread
            self.voice_recording_thread = threading.Thread(target=self.voice_recording_loop)
            self.voice_recording_thread.daemon = True
            self.voice_recording_thread.start()
            
        except Exception as e:
            error_msg = f"Failed to start voice recording: {str(e)}"
            print(error_msg)
            messagebox.showerror("Voice Input Error", 
                f"{error_msg}\n\nPlease check:\n- Microphone is connected\n- Microphone permissions are granted\n- No other apps are using the microphone")
            self.is_voice_recording = False
            self.voice_button.configure(text="[MIC] Start Voice Input")
            self.voice_status_label.configure(text="Voice: Error", text_color="#e74c3c")
    
    def stop_voice_recording(self):
        """Stop voice recording"""
        self.is_voice_recording = False
        self.voice_button.configure(text="[MIC] Start Voice Input")
        self.voice_status_label.configure(text="Voice: Stopping...", text_color="#f39c12")
        
        # Wait for recording thread to finish
        if self.voice_recording_thread and self.voice_recording_thread.is_alive():
            self.voice_recording_thread.join(timeout=2)
        
        self.voice_status_label.configure(text="Voice: Ready", text_color="#2ecc71")
    
    def voice_recording_loop(self):
        """Main voice recording loop with improved error handling and functionality"""
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise with shorter duration for faster startup
                self.recognition_engine.adjust_for_ambient_noise(source, duration=0.5)
                
                # Update status to show we're ready to listen
                self.root.after(0, lambda: self.voice_status_label.configure(text="Voice: Listening...", text_color="#f39c12"))
                
                while self.is_voice_recording:
                    try:
                        # Listen for audio with shorter timeout for more responsive feel
                        audio = self.recognition_engine.listen(source, timeout=0.5, phrase_time_limit=8)
                        
                        # Update status to show we're processing
                        self.root.after(0, lambda: self.voice_status_label.configure(text="Voice: Processing...", text_color="#3498db"))
                        
                        # Recognize speech using Google Speech Recognition
                        text = self.recognition_engine.recognize_google(audio)
                        
                        if text and text.strip():
                            # Update UI on main thread
                            def update_text():
                                try:
                                    current = self.text_input.get("1.0", "end-1c")
                                    if current == "Type how you feel or use voice input to express your emotions...":
                                        self.text_input.delete("1.0", "end")
                                        self.text_input.configure(text_color=TEXT_COLOR)
                                        current = ""
                                    
                                    # Add transcribed text with timestamp
                                    timestamp = datetime.datetime.now().strftime("%H:%M:%S") 
                                    spacer = "\n" if current and not current.endswith("\n") else ""
                                    self.text_input.insert("end", f"{spacer}[{timestamp}] Voice: {text.strip()}")
                                    self.text_input.see("end")
                                    self.update_text_analysis()
                                    
                                    # Update status with success message
                                    display_text = text.strip()[:30] + "..." if len(text.strip()) > 30 else text.strip()
                                    self.voice_status_label.configure(text=f"Voice: OK '{display_text}'", text_color="#2ecc71")
                                except Exception as e:
                                    print(f"Error updating text from voice: {e}")
                            
                            self.root.after(0, update_text)
                            
                    except sr.WaitTimeoutError:
                        # No speech detected, continue listening
                        continue
                    except sr.UnknownValueError:
                        # Speech was unintelligible - show brief error then continue
                        self.root.after(0, lambda: self.voice_status_label.configure(text="Voice: Could not understand - try again", text_color="#f39c12"))
                        time.sleep(1)  # Brief pause before continuing
                        continue
                    except sr.RequestError as e:
                        # API was unreachable or unresponsive
                        self.root.after(0, lambda: self.voice_status_label.configure(text="Voice: Network error - check internet", text_color="#e74c3c"))
                        time.sleep(2)  # Longer pause for network issues
                        continue
                    except Exception as e:
                        # Other errors
                        print(f"Voice recognition error: {e}")
                        self.root.after(0, lambda: self.voice_status_label.configure(text="Voice: Error occurred", text_color="#e74c3c"))
                        time.sleep(1)
                        continue
                        
        except Exception as e:
            error_msg = f"Voice recording setup error: {str(e)}"
            print(error_msg)
            error_str = str(e)  # Capture error message explicitly
            self.root.after(0, lambda msg=error_str: messagebox.showerror("Voice Input Error", 
                f"Could not initialize voice recording.\n\nError: {msg}\n\nPlease check:\n- Microphone permissions\n- Internet connection\n- No other apps using microphone"))
            self.root.after(0, lambda: self.voice_status_label.configure(text="Voice: Setup failed", text_color="#e74c3c"))
    
    def start_audio_visualization(self):
        """Start audio level visualization with safety checks"""
        try:
            print("[SAFETY] Starting audio visualization...")
            if not PYAUDIO_AVAILABLE or pyaudio is None:
                # Audio visualization not available, show message
                print("[INFO] PyAudio not available, audio visualization disabled")
                if hasattr(self, 'audio_level_label'):
                    self.audio_level_label.config(text="Audio: Not Available", fg="#e74c3c")
                return
            
            self.audio_visualization_thread = threading.Thread(target=self.audio_visualization_loop)
            self.audio_visualization_thread.daemon = True
            self.audio_visualization_thread.start()
            print("[OK] Audio visualization started successfully")
        except Exception as e:
            print(f"[ERROR] Failed to start audio visualization: {e}")
            if hasattr(self, 'audio_level_label'):
                self.audio_level_label.configure(text="Audio: Error", text_color="#e74c3c")

    def start_audio_processing(self):
        """Initialize a single shared PyAudio stream and start reader thread"""
        if not PYAUDIO_AVAILABLE or pyaudio is None:
            print("[INFO] PyAudio not available, skipping shared audio processing setup")
            return
        try:
            if self.pyaudio_instance is None:
                self.pyaudio_instance = pyaudio.PyAudio()
            # Configure shared stream parameters
            self.audio_chunk_size = 1024  # keep small for UI responsiveness
            self.audio_rate = 44100
            if self.audio_stream is None:
                self.audio_stream = self.pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.audio_rate,
                    input=True,
                    frames_per_buffer=self.audio_chunk_size
                )
            # Start reader thread once
            if not self.audio_reader_thread or not self.audio_reader_thread.is_alive():
                self.is_audio_reader_running = True
                self.audio_reader_thread = threading.Thread(target=self.audio_reader_loop, name="audio_reader_loop")
                self.audio_reader_thread.daemon = True
                self.audio_reader_thread.start()
                print("[OK] Shared audio reader started")
        except Exception as e:
            print(f"[ERROR] Failed to start shared audio processing: {e}")
            try:
                if self.audio_stream:
                    self.audio_stream.close()
                    self.audio_stream = None
                if self.pyaudio_instance:
                    self.pyaudio_instance.terminate()
                    self.pyaudio_instance = None
            except Exception:
                pass

    def audio_reader_loop(self):
        """Continuously read from the shared stream and feed audio_queue"""
        print(f"[DEBUG] Starting Audio Reader Loop. Stream: {self.audio_stream}")
        if not PYAUDIO_AVAILABLE or pyaudio is None or self.audio_stream is None:
            print("[DEBUG] Audio stream not available")
            return
        
        while self.is_audio_reader_running:
            try:
                if self.audio_stream.is_active():
                    data = self.audio_stream.read(self.audio_chunk_size, exception_on_overflow=False)
                    # [DEBUG] Print periodically if data is being read (e.g. random sample)
                    if np.random.random() < 0.01:
                         print(f"[DEBUG] Audio chunk captured, size: {len(data)}")
                    
                    # Keep the most recent chunks; drop oldest if full
                    try:
                        self.audio_queue.put_nowait(data)
                    except queue.Full:
                        try:
                            _ = self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait(data)
                        except Exception:
                            pass
                else:
                    print("[DEBUG] Audio stream is inactive")
                    time.sleep(0.1)
            except (IOError, OSError) as audio_err:
                print(f"[WARN] Shared audio read error: {audio_err}. Retrying...")
                time.sleep(0.5)
            except Exception as e:
                print(f"[ERROR] audio_reader_loop: {e}")
                time.sleep(0.5)
    
    def audio_visualization_loop(self):
        """Audio level visualization loop"""
        if not PYAUDIO_AVAILABLE or pyaudio is None:
            return
            
        try:
            # Consume shared audio chunks from queue (no separate stream)
            CHUNK = self.audio_chunk_size if hasattr(self, 'audio_chunk_size') else 1024
            RATE = self.audio_rate if hasattr(self, 'audio_rate') else 44100
            while getattr(self, 'is_audio_visualization_running', True):
                try:
                    # Read audio data from shared queue
                    try:
                        data = self.audio_queue.get(timeout=1)
                    except Exception:
                        # No data available; continue
                        time.sleep(0.1)
                        continue

                    # Convert to numpy array and calculate RMS
                    audio_data_int = np.frombuffer(data, dtype=np.int16)
                    # Handle empty or invalid audio data
                    if len(audio_data_int) > 0 and np.any(audio_data_int != 0):
                        rms = np.sqrt(np.mean(audio_data_int.astype(np.float32)**2))
                    else:
                        rms = 0
                    
                    # Store audio data for emotion analysis (normalized)
                    if len(audio_data_int) > 0:
                        audio_normalized = audio_data_int.astype(np.float32) / 32768.0
                        # Add to buffer for emotion analysis
                        if not hasattr(self, 'audio_emotion_buffer'):
                            self.audio_emotion_buffer = []
                        self.audio_emotion_buffer.append(audio_normalized)
                        # Keep only last 3 seconds of audio
                        max_buffer_size = int(RATE * 3 / CHUNK)  # 3 seconds
                        if len(self.audio_emotion_buffer) > max_buffer_size:
                            self.audio_emotion_buffer = self.audio_emotion_buffer[-max_buffer_size:]
                    
                    # Convert to decibels (approximate)
                    if rms > 0:
                        db = 20 * np.log10(rms / 32768.0)
                        db = max(-60, min(0, db))  # Clamp between -60 and 0 dB
                    else:
                        db = -60
                    
                    # *** THREAD-SAFE FIX ***
                    # Update UI on main thread - capture values to avoid closure issues
                    db_value = db
                    rms_value = rms
                    def update_audio_level():
                        try:
                            if hasattr(self, 'root') and hasattr(self, 'audio_level_label'):
                                self.audio_level_label.configure(text=f"Audio Level: {db_value:.1f} dB")
                                self.update_audio_waveform(rms_value)
                        except (tk.TclError, AttributeError, RuntimeError):
                            pass  # Window already destroyed or being destroyed
                    
                    try:
                        if hasattr(self, 'root'):
                            self.root.after(0, update_audio_level)
                    except (tk.TclError, AttributeError, RuntimeError):
                        break  # Window destroyed, exit loop
                    
                    time.sleep(0.1)  # Update every 100ms
                    
                except Exception as e:
                    # This block makes the loop robust
                    print(f"Audio visualization error: {e}")
                    time.sleep(1) # Sleep to prevent spamming errors
            
            # Clean up
            try:
                if 'stream' in locals():
                    stream.stop_stream()
                    stream.close()
                if 'p' in locals():
                    p.terminate()
            except Exception as cleanup_error:
                print(f"Error during audio cleanup: {cleanup_error}")
            
        except Exception as e:
            print(f"Failed to initialize audio visualization: {e}")
            # Show user-friendly error message
            try:
                if hasattr(self, 'audio_level_label'):
                    self.root.after(0, lambda: self.audio_level_label.config(text="Audio: Not Available", fg="#e74c3c"))
            except Exception:
                pass
    
    def update_audio_waveform(self, rms):
        """Update the audio waveform visualization"""
        # This function is now only called from the main thread
        try:
            # Clear canvas
            self.audio_canvas.delete("all")
            
            # Get canvas dimensions
            width = self.audio_canvas.winfo_width()
            height = self.audio_canvas.winfo_height()
            
            if width <= 1 or height <= 1:
                return
            
            # Normalize RMS to 0-1 range
            normalized_rms = min(1.0, rms / 10000.0)
            
            # Draw waveform bars
            bar_width = width // 20
            bar_spacing = 2
            
            for i in range(20):
                x = i * (bar_width + bar_spacing)
                bar_height = int(height * normalized_rms * (0.5 + 0.5 * np.sin(i * 0.5)))
                
                # Color based on intensity
                if normalized_rms > 0.7:
                    color = "#e74c3c"  # Red for high intensity
                elif normalized_rms > 0.4:
                    color = "#f39c12"  # Orange for medium intensity
                else:
                    color = "#2ecc71"  # Green for low intensity
                
                # Draw bar
                self.audio_canvas.create_rectangle(
                    x, height - bar_height,
                    x + bar_width, height,
                    fill=color, outline=""
                )
                
        except Exception as e:
            print(f"Waveform update error: {e}")

    def update_text_analysis(self, event=None):
        """Analyze text emotion when typing"""
        try:
            # This is called by a tkinter event, so it's already on the main thread
            text_content = self.text_input.get("1.0", "end-1c").strip()
            
            # Only analyze if text has changed significantly
            if text_content and text_content != self.last_text_input:
                self.last_text_input = text_content
                self.last_update_time = datetime.datetime.now()
                
                # Analyze text emotions using Text Module
                self.text_emotions = self.text_analyzer.analyze(text_content)
                
                # Update text emotion bars with improved visual feedback
                if any(v > 0 for v in self.text_emotions.values()):
                    dominant_emotion = max(self.text_emotions, key=self.text_emotions.get)
                    
                    if hasattr(self, 'text_emotion_bars'):
                        for emotion, value in self.text_emotions.items():
                            if emotion in self.text_emotion_bars:
                                self.text_emotion_bars[emotion].set(value)
                                
                                # Highlight dominant emotion
                                if emotion == dominant_emotion:
                                    self.text_emotion_bars[emotion].configure(progress_color=self.get_emotion_color(emotion))
                                else:
                                    self.text_emotion_bars[emotion].configure(progress_color="#3498db")
        except Exception as e:
            print(f"Error in text analysis: {e}")

    def start_audio_analysis(self):
        """Start audio analysis with safety checks"""
        try:
            print("[SAFETY] Starting audio analysis...")
            self.is_audio_running = True
            self.audio_thread = threading.Thread(target=self.audio_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            print("[OK] Audio analysis started successfully")
        except Exception as e:
            print(f"[ERROR] Failed to start audio analysis: {e}")
            self.is_audio_running = False

    def start_transcription_if_available(self):
        """Start transcription with safety checks"""
        # This transcription loop is separate from the voice input
        # and seems to be for continuous background transcription.
        # It's already thread-safe using root.after().
        try:
            print("[SAFETY] Starting transcription...")
            if not SPEECH_AVAILABLE or sr is None:
                print("[INFO] Speech recognition not available, skipping transcription")
                return
            self.is_transcribing = True
            self.transcription_thread = threading.Thread(target=self.transcription_loop)
            self.transcription_thread.daemon = True
            self.transcription_thread.start()
            print("[OK] Transcription started successfully")
        except Exception as e:
            print(f"[ERROR] Failed to start transcription: {e}")
            self.is_transcribing = False
            
    def capture_audio_emotion(self):
        """Captures the current audio emotion on button press"""
        # This is called by a button, so it's on the main thread
        # Store the current audio emotions
        self.captured_audio_emotions = self.audio_emotions.copy()
        
        # Update the UI to show captured state
        self.update_audio_emotion_display(use_captured=True)
        
        # Show confirmation message
        self.show_capture_confirmation("Voice emotion captured successfully!")

    def update_audio_emotion_display(self, use_captured=False):
        """Updates the audio emotion dashboard with visual feedback for dominant emotion."""
        # This function is now only called from the main thread
        try:
            # Find dominant audio emotion
            emotions_to_display = self.captured_audio_emotions if use_captured and hasattr(self, 'captured_audio_emotions') else self.audio_emotions
            
            if any(v > 0 for v in emotions_to_display.values()):
                dominant_emotion = max(emotions_to_display, key=emotions_to_display.get)
                
                # Update progress bars with visual feedback for dominant emotion
                for emotion, value in emotions_to_display.items():
                    if emotion in self.audio_emotion_bars:
                         # Set progress bar value
                        self.audio_emotion_bars[emotion].set(value)
                        
                        # Highlight only the dominant emotion
                        if emotion == dominant_emotion:
                            self.audio_emotion_bars[emotion].configure(progress_color=self.get_emotion_color(emotion))
                        else:
                            self.audio_emotion_bars[emotion].configure(progress_color="#3498db")
            else:
                # Reset all progress bars if no emotions detected
                for emotion in EMOTIONS:
                    if emotion in self.audio_emotion_bars:
                        self.audio_emotion_bars[emotion].set(0)
                        self.audio_emotion_bars[emotion].configure(progress_color="#3498db")
        except Exception as e:
            print(f"Error updating audio display: {e}")
                
    def audio_loop(self):
        """Audio emotion analysis loop that captures and analyzes audio using Audio Module"""
        print("[INFO] Starting audio processing loop")
        # Initialize Audio Module capabilities check
        if not self.audio_analyzer:
             print("[WARN] Audio analyzer not initialized")
             return

        try:
            CHUNK = self.audio_chunk_size
            RATE = self.audio_rate
            audio_buffer = []
            buffer_duration = 3  # Analyze 3 seconds of audio
            samples_per_analysis = int(RATE * buffer_duration / CHUNK)
            chunk_count = 0

            while self.is_audio_running:
                try:
                    # Pull audio chunk from shared queue
                    try:
                        data = self.audio_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    except Exception:
                        time.sleep(0.1)
                        continue

                    # Convert to float32
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                    # Add to buffer
                    audio_buffer.append(audio_data)
                    chunk_count += 1

                    # Analyze when we have enough data
                    if chunk_count >= samples_per_analysis:
                        full_audio = np.concatenate(audio_buffer)
                        
                        # Analyze using Audio Module
                        self.audio_emotions = self.audio_analyzer.analyze(
                            audio_data=full_audio,
                            sample_rate=RATE
                        )
                        
                        # Schedule the GUI update on the main thread
                        self.root.after(0, self.update_audio_emotion_display)
                        
                        # Reset buffer
                        audio_buffer = []
                        chunk_count = 0

                    time.sleep(0.01) # High performance loop
                except Exception as e:
                    print(f"Error in audio_loop: {e}")
                    time.sleep(0.5)
        except Exception as e:
            print(f"Error initializing audio loop: {e}")
            # Fallback behavior is handled by the AudioAnalyzer internal logic
            pass

    def transcription_loop(self):
        # This function is already thread-safe as it uses root.after()
        if not SPEECH_AVAILABLE or sr is None:
            return
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
                while getattr(self, 'is_transcribing', False):
                    try:
                        audio = recognizer.listen(source, timeout=2, phrase_time_limit=5)
                        text = recognizer.recognize_google(audio)
                        if text:
                            # Append to text input safely on UI thread
                            def append_text():
                                try:
                                    current = self.text_input.get("1.0", "end-1c")
                                    spacer = "\n" if current and not current.endswith("\n") else ""
                                    self.text_input.insert("end", f"{spacer}{text}")
                                    self.text_input.see("end")
                                    self.update_text_analysis()
                                except Exception as e:
                                    print(f"Error appending transcript: {e}")
                            self.root.after(0, append_text)
                    except sr.WaitTimeoutError:
                        continue
                    except Exception:
                        continue
        except Exception as e:
            print(f"Error in transcription_loop setup: {e}")
            pass

    def update_emotion_display(self):
        """Updates the emotion display with combined emotion scores using modular Fusion."""
        # This function runs on the main thread via root.after()
        try:
            # Check if window still exists
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                return
                
            current_time = datetime.datetime.now()
            
            # --- 1. Confidence Estimation ---
            # Define confidences based on input activity
            visual_conf = 1.0 if (self.is_camera_running and self.face_detected) else 0.0
            
            text_content = self.text_input.get("1.0", "end-1c").strip()
            text_active = text_content and text_content != "Type how you feel or use voice input to express your emotions..."
            text_conf = 1.0 if text_active else 0.0
            
            audio_conf = 0.8 if self.is_audio_running else 0.0 
            
            # --- 2. Multimodal Fusion ---
            # Use Fusion Module
            combined_emotions = self.fusion.fuse(
                self.visual_emotions,
                self.audio_emotions,
                self.text_emotions,
                visual_conf=visual_conf,
                audio_conf=audio_conf,
                text_conf=text_conf
            )
            
            self.combined_emotions = combined_emotions
            
            # --- 3. Update History ---
            self.timestamps.append(current_time)
            for emotion, score in combined_emotions.items():
                self.emotion_history[emotion].append(score)
                
            # Keep history manageable
            if len(self.timestamps) > 100:
                self.timestamps.pop(0)
                for emotion in EMOTIONS:
                    self.emotion_history[emotion].pop(0)

            # --- 4. Update UI Elements ---
            if any(v > 0 for v in combined_emotions.values()):
                dominant_emotion = max(combined_emotions, key=combined_emotions.get)
                confidence = combined_emotions[dominant_emotion]
                
                # Get Color
                color = self.get_emotion_color(dominant_emotion)
                
                # Update Labels
                if hasattr(self, 'dominant_emotion_label'):
                    self.dominant_emotion_label.configure(
                        text=f"{dominant_emotion.capitalize()}",
                        text_color=color
                    )
                
                if hasattr(self, 'confidence_label'):
                    self.confidence_label.configure(
                        text=f"Confidence: {confidence:.2f}",
                        text_color="#2ecc71" if confidence > 0.5 else "#f39c12"
                    )
                    
                # Update Recommendation (every 10th frame)
                if self.analysis_count % 10 == 0:
                     if hasattr(self, 'recommendation_label'):
                        rec_text = self.recommender.get_recommendation(
                            dominant_emotion,
                            self.emotion_history,
                            current_time
                        )
                        self.recommendation_label.configure(text=rec_text)

                # Update Progress Bars
                if hasattr(self, 'facial_emotion_bars'):
                    for emotion, bar in self.facial_emotion_bars.items():
                        bar.set(self.visual_emotions.get(emotion, 0.0))
                
                if hasattr(self, 'text_emotion_bars'):
                    for emotion, bar in self.text_emotion_bars.items():
                        bar.set(self.text_emotions.get(emotion, 0.0))
                        
                if hasattr(self, 'audio_emotion_bars'):
                    for emotion, bar in self.audio_emotion_bars.items():
                        bar.set(self.audio_emotions.get(emotion, 0.0))

            # Update Analytics (every 10th frame)
            self.analysis_count += 1
            if self.analysis_count % 10 == 0:
                self.update_analytics()

        except Exception as e:
            print(f"Error in update_emotion_display: {e}")
            import traceback
            traceback.print_exc()

        # Schedule the next update
        try:
            self.update_timer = self.root.after(100, self.update_emotion_display)
        except Exception:
            pass

    def get_emotion_color(self, emotion):
        """Return color code for emotion"""
        colors = {
            "happy": "#f1c40f", "sad": "#3498db", "angry": "#e74c3c",
            "fear": "#9b59b6", "disgust": "#2ecc71", "surprise": "#e67e22",
            "neutral": "#95a5a6"
        }
        return colors.get(emotion, "#ecf0f1")

    def update_analytics(self):
        """Updates the analytical dashboard with current emotion data"""
        if not self.timestamps or len(self.timestamps) < 2:
            return
        
        # Check if matplotlib is available
        if not hasattr(self, 'matplotlib_available') or not self.matplotlib_available:
            return
        
        if self.fig is None or self.ax is None or self.canvas is None:
            return
            
        # Calculate emotional stability score
        stability_score = self.calculate_stability_score()
        self.stability_score_label.config(text=f"{stability_score:.1f}%")
        self.stability_progress['value'] = stability_score
        
        # Clear the canvas
        self.ax.clear()
        
        # Format timestamps for x-axis
        formatted_times = [t.strftime('%H:%M:%S') for t in self.timestamps]
        
        # Plot each emotion as a line with improved styling
        emotion_colors = {
            "happy": "#2ecc71",  # Green
            "sad": "#3498db",    # Blue
            "angry": "#e74c3c",  # Red
            "fear": "#9b59b6",   # Purple
            "disgust": "#e67e22", # Orange
            "surprise": "#f39c12", # Yellow
            "neutral": "#95a5a6"  # Gray
        }
        
        # Find dominant emotion for each time point
        dominant_emotions = []
        for i in range(len(self.timestamps)):
            # Safely get emotions at this time point
            emotions_at_time = {}
            for emotion in EMOTIONS:
                if emotion in self.emotion_history and i < len(self.emotion_history[emotion]):
                    emotions_at_time[emotion] = self.emotion_history[emotion][i]
                else:
                    emotions_at_time[emotion] = 0.0
            if emotions_at_time:
                dominant = max(emotions_at_time, key=emotions_at_time.get)
                dominant_emotions.append(dominant)
            else:
                dominant_emotions.append("neutral")
        
        # Draw lines for each emotion
        for emotion in EMOTIONS:
            if len(self.timestamps) > 0 and emotion in self.emotion_history and len(self.emotion_history[emotion]) > 0:
                color = emotion_colors.get(emotion, "#3498db")
                
                # Get points for this emotion
                points = []
                for i in range(len(self.timestamps)):
                    if i < len(self.emotion_history[emotion]):
                        points.append((i, self.emotion_history[emotion][i]))
                
                # Plot with different line styles based on dominance
                x_vals = [p[0] for p in points]
                y_vals = [p[1] for p in points]
                
                # Plot with thicker line for dominant emotions
                is_dominant = [dominant_emotions[i] == emotion for i in range(len(dominant_emotions)) if i < len(x_vals)]
                
                # Plot regular line
                self.ax.plot(
                    x_vals, 
                    y_vals, 
                    label=emotion.capitalize(),
                    color=color,
                    marker='o',
                    markersize=4,
                    linewidth=1,
                    alpha=0.6
                )
                
                # Highlight dominant segments with thicker lines
                for i in range(len(x_vals)-1):
                    if i < len(is_dominant) and is_dominant[i]:
                        self.ax.plot(
                            [x_vals[i], x_vals[i+1]], 
                            [y_vals[i], y_vals[i+1]],
                            color=color,
                            linewidth=3,
                            alpha=1.0
                        )
        
        # Configure the chart with improved styling
        self.ax.set_title("Multimodal Emotion History", color=TEXT_COLOR, fontsize=12, fontweight='bold')
        self.ax.set_xlabel("Time", color=TEXT_COLOR, fontsize=10)
        self.ax.set_ylabel("Intensity", color=TEXT_COLOR, fontsize=10)
        self.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        
        # Use time labels for x-axis
        if len(formatted_times) > 5:
            # Show only a few time labels to avoid overcrowding
            step = max(1, len(formatted_times) // 5)
            self.ax.set_xticks(range(0, len(formatted_times), step))
            self.ax.set_xticklabels([formatted_times[i] for i in range(0, len(formatted_times), step)])
        else:
            self.ax.set_xticks(range(len(formatted_times)))
            self.ax.set_xticklabels(formatted_times)
        
        # Improve legend
        self.ax.legend(
            loc='upper right', 
            facecolor=BACKGROUND_COLOR, 
            edgecolor='gray',
            framealpha=0.8,
            fontsize=8
        )
        
        # Add grid for better readability
        self.ax.grid(True, linestyle='--', alpha=0.3)
        
        # Rotate x-axis labels for better readability
        try:
            plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')
            
            # Update the canvas with tight layout
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating analytics plot: {e}")
    
    def calculate_stability_score(self):
        """Calculate emotional stability score based on emotion history."""
        if not self.emotion_history:
            return 50.0  # Default score
        
        # Check if we have any data points
        total_points = sum(len(values) for values in self.emotion_history.values())
        if total_points == 0:
            return 50.0  # Default score
        
        # Calculate variance of emotions over time
        variances = []
        for emotion in EMOTIONS:
            if emotion in self.emotion_history and len(self.emotion_history[emotion]) > 1:
                # Calculate variance of this emotion over time
                values = self.emotion_history[emotion]
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                variances.append(variance)
        
        if not variances:
            return 50.0
            
        # Average variance across all emotions
        avg_variance = sum(variances) / len(variances)
        
        # Convert variance to stability score (lower variance = higher stability)
        # Scale between 0-100, where 100 is perfectly stable
        stability = 100 - (avg_variance * 500)  # Scaling factor
        
        # Ensure score is within bounds
        return max(0, min(100, stability))
            
    def on_closing(self):
        print("Closing application...")
        # Set flags to stop all threads
        self.is_camera_running = False
        self.is_audio_running = False
        self.is_audio_reader_running = False
        self.is_transcribing = False
        self.is_voice_recording = False
        self.is_audio_visualization_running = False

        # Stop the main UI update loop
        if self.update_timer:
            try:
                self.root.after_cancel(self.update_timer)
            except Exception:
                pass
            self.update_timer = None
        
        # Wait for all threads to finish
        threads_to_join = [
            self.video_thread,
            self.audio_thread, 
            self.transcription_thread,
            self.voice_recording_thread,
            getattr(self, 'audio_visualization_thread', None),
            getattr(self, 'audio_reader_thread', None)
        ]
        
        for thread in threads_to_join:
            if thread and thread.is_alive():
                print(f"Waiting for {thread.name} to join...")
                thread.join(timeout=0.5)
        
        print("All threads joined.")

        # Cleanup shared audio stream
        try:
            if getattr(self, 'audio_stream', None):
                try:
                    self.audio_stream.stop_stream()
                except Exception:
                    pass
                try:
                    self.audio_stream.close()
                except Exception:
                    pass
                self.audio_stream = None
            if getattr(self, 'pyaudio_instance', None):
                try:
                    self.pyaudio_instance.terminate()
                except Exception:
                    pass
                self.pyaudio_instance = None
        except Exception:
            pass

        # Cleanup new modules
        try:
             # Audio Analyzer (might have pyAudio stream)
            if hasattr(self, 'audio_analyzer') and hasattr(self.audio_analyzer, 'cleanup'):
                 self.audio_analyzer.cleanup()
        except Exception:
             pass
             
        try:
            self.root.destroy()
        except Exception:
            pass
