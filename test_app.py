#!/usr/bin/env python
"""Quick test to verify the application can start"""

import sys
import tkinter as tk

try:
    from main import EmotionRecommenderApp
    print("✓ Successfully imported EmotionRecommenderApp")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

try:
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    print("✓ Created tkinter root window")
    
    app = EmotionRecommenderApp(root)
    print("✓ Successfully created EmotionRecommenderApp instance")
    
    # Clean up
    root.destroy()
    print("✓ Application test passed!")
    
except Exception as e:
    print(f"✗ Error during initialization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed! Application is ready to run.")
print("Run 'python main.py' to start the application.")

