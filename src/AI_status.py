import tkinter as tk

PURPLE = "#5B2D8E"
YELLOW = "#F5A623"

root = tk.Tk()
root.title("AI Status")
root.configure(bg=PURPLE)
root.geometry("400x150")

tk.Label(root, text="⚡ smarti", font=("Arial", 24, "bold"),
         bg=PURPLE, fg=YELLOW).pack(pady=(20,5))

tk.Label(root, text="AI Monitoring Active", font=("Arial", 14, "bold"),
         bg=PURPLE, fg="white").pack()

tk.Label(root, text="Smile, you are being recorded", font=("Arial", 11),
         bg=PURPLE, fg="#DDDDDD").pack()

root.mainloop()