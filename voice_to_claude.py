"""
Voice Input для Claude Code — трей-приложение
----------------------------------------------
Клик по иконке → окно настроек.
Поддерживает клавиши клавиатуры и кнопки мыши (X1, X2, Middle и др.)
Config: {"hotkey": "f9"} или {"hotkey": "mouse:x1"}
"""

import json
import os
import threading
import time
import queue
import tkinter as tk
from tkinter import ttk, font as tkfont
import numpy as np
import sounddevice as sd
import keyboard
import pyperclip
import pystray
from PIL import Image, ImageDraw
from pynput import mouse as pynput_mouse

import sys
_base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(_base_dir, "config.json")
SAMPLE_RATE = 16000

# ── Состояние ────────────────────────────────────────────────────────────────
is_recording = False
audio_frames = []
stream = None
model = None
lock = threading.Lock()
tray_icon = None
current_hotkey_handle = None
current_mouse_listener = None
tk_queue = queue.Queue()
settings_open = False

# Имена кнопок мыши
MOUSE_BUTTON_NAMES = {
    "mouse:x1":     "Мышь X1 (назад)",
    "mouse:x2":     "Мышь X2 (вперёд)",
    "mouse:middle": "Мышь (колёсико)",
}

PYNPUT_BUTTON_MAP = {
    "mouse:x1":     pynput_mouse.Button.x1,
    "mouse:x2":     pynput_mouse.Button.x2,
    "mouse:middle": pynput_mouse.Button.middle,
}


# ── Конфиг ───────────────────────────────────────────────────────────────────

def load_config():
    defaults = {"hotkey": "f9", "model": "tiny", "language": "ru"}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        defaults.update(cfg)
    return defaults


def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def hotkey_display_name(hotkey_str):
    """Читаемое имя горячей клавиши."""
    return MOUSE_BUTTON_NAMES.get(hotkey_str, hotkey_str.upper())


# ── Иконки ───────────────────────────────────────────────────────────────────

def make_icon(bg_color):
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([2, 2, size - 2, size - 2], fill=bg_color)
    white = (255, 255, 255, 230)
    d.rounded_rectangle([22, 10, 42, 38], radius=8, fill=white)
    d.arc([16, 26, 48, 50], start=0, end=180, fill=white, width=4)
    d.line([32, 50, 32, 58], fill=white, width=4)
    d.line([24, 58, 40, 58], fill=white, width=4)
    return img


ICON_IDLE = make_icon((80, 80, 80, 220))
ICON_REC  = make_icon((200, 40, 40, 220))
ICON_PROC = make_icon((200, 160, 0, 220))
ICON_OK   = make_icon((40, 160, 40, 220))


def set_icon(icon_img, tooltip=None):
    if tray_icon:
        tray_icon.icon = icon_img
        if tooltip:
            tray_icon.title = tooltip


# ── Загрузка модели ───────────────────────────────────────────────────────────

def load_model_bg(model_name):
    global model
    from faster_whisper import WhisperModel
    # Добавляем CUDA DLL в PATH
    import site
    sp = site.getsitepackages()[0] if hasattr(site, 'getsitepackages') else os.path.join(os.path.dirname(os.__file__), 'site-packages')
    for pkg in ('cublas', 'cudnn', 'cuda_nvrtc'):
        dll_dir = os.path.join(sp, 'nvidia', pkg, 'bin')
        if os.path.isdir(dll_dir):
            os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')
    set_icon(ICON_PROC, "Voice Input: загрузка модели...")
    try:
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
    except Exception:
        model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=16)
    cfg = load_config()
    set_icon(ICON_IDLE, f"Voice Input: готово  [{hotkey_display_name(cfg['hotkey'])}]")


# ── Запись и транскрибция ─────────────────────────────────────────────────────

def audio_callback(indata, frames, time_info, status):
    if is_recording:
        audio_frames.append(indata.copy())


def start_recording():
    global stream, audio_frames, is_recording
    audio_frames = []
    is_recording = True
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        callback=audio_callback, blocksize=1024,
    )
    stream.start()
    set_icon(ICON_REC, "Voice Input: запись...")


def stop_and_transcribe(language):
    global stream, is_recording
    is_recording = False
    if stream:
        stream.stop()
        stream.close()
        stream = None

    if not audio_frames:
        set_icon(ICON_IDLE, "Voice Input: пустая запись")
        return

    set_icon(ICON_PROC, "Voice Input: обработка...")
    audio = np.concatenate(audio_frames).flatten()

    lang = language if language and language != "null" else None
    segments, _ = model.transcribe(audio, language=lang, beam_size=5)
    text = " ".join(seg.text for seg in segments).strip()

    # Фильтр галлюцинаций Whisper
    HALLUCINATIONS = {"музыкальная заставка", "динамичная музыка", "субтитры", "продолжение следует",
                      "подписывайтесь на канал", "thanks for watching", "music", "applause"}
    if text.lower().strip("., !") in HALLUCINATIONS:
        set_icon(ICON_IDLE, "Voice Input: шум, пропуск")
        return

    if not text:
        set_icon(ICON_IDLE, "Voice Input: ничего не распознано")
        return

    pyperclip.copy(text)
    time.sleep(0.3)
    keyboard.send("ctrl+v")

    set_icon(ICON_OK, f"✓ {text[:60]}")
    time.sleep(1.5)
    cfg = load_config()
    set_icon(ICON_IDLE, f"Voice Input: готово  [{hotkey_display_name(cfg['hotkey'])}]")


def on_hotkey():
    if model is None:
        set_icon(ICON_PROC, "Voice Input: модель ещё грузится...")
        return
    cfg = load_config()
    with lock:
        if not is_recording:
            start_recording()
        else:
            threading.Thread(
                target=stop_and_transcribe,
                args=(cfg.get("language", "ru"),),
                daemon=True
            ).start()


# ── Регистрация горячей клавиши (клавиатура или мышь) ────────────────────────

def unregister_hotkey():
    global current_hotkey_handle, current_mouse_listener
    if current_hotkey_handle is not None:
        try:
            keyboard.remove_hotkey(current_hotkey_handle)
        except Exception:
            pass
        current_hotkey_handle = None
    if current_mouse_listener is not None:
        try:
            current_mouse_listener.stop()
        except Exception:
            pass
        current_mouse_listener = None


def register_hotkey(hotkey_str):
    global current_hotkey_handle, current_mouse_listener
    unregister_hotkey()

    if hotkey_str.startswith("mouse:"):
        btn = PYNPUT_BUTTON_MAP.get(hotkey_str)
        if btn is None:
            return

        # X1 (ближняя боковая) = Enter
        enter_btn = pynput_mouse.Button.x1

        def on_click(x, y, button, pressed):
            if pressed and button == btn:
                on_hotkey()
            elif pressed and button == enter_btn and btn != enter_btn:
                keyboard.send("enter")

        listener = pynput_mouse.Listener(on_click=on_click)
        listener.start()
        current_mouse_listener = listener
    else:
        current_hotkey_handle = keyboard.add_hotkey(hotkey_str, on_hotkey, suppress=False)


# ── Окно настроек (tkinter) ───────────────────────────────────────────────────

def open_settings_window(tk_root):
    global settings_open
    if settings_open:
        return
    settings_open = True

    cfg = load_config()

    win = tk.Toplevel(tk_root)
    win.title("Voice Input — Настройки")
    win.resizable(False, False)
    win.attributes("-topmost", True)
    win.grab_set()

    w, h = 420, 320
    sx = win.winfo_screenwidth()
    sy = win.winfo_screenheight()
    win.geometry(f"{w}x{h}+{(sx - w) // 2}+{(sy - h) // 2}")

    PAD  = {"padx": 16, "pady": 6}
    BG   = "#1e1e2e"
    FG   = "#cdd6f4"
    ACC  = "#89b4fa"
    BTN  = "#313244"
    BTN_H = "#45475a"

    win.configure(bg=BG)
    title_f = tkfont.Font(family="Segoe UI", size=13, weight="bold")
    label_f = tkfont.Font(family="Segoe UI", size=10)
    btn_f   = tkfont.Font(family="Segoe UI", size=10, weight="bold")

    # ── Заголовок
    tk.Label(win, text="Настройки Voice Input", font=title_f,
             bg=BG, fg=FG).pack(pady=(18, 4))
    tk.Frame(win, bg=ACC, height=1).pack(fill="x", padx=16, pady=(0, 10))

    # ── Горячая клавиша
    hk_frame = tk.Frame(win, bg=BG)
    hk_frame.pack(fill="x", **PAD)
    tk.Label(hk_frame, text="Горячая клавиша:", font=label_f,
             bg=BG, fg=FG, width=18, anchor="w").pack(side="left")

    current_display = hotkey_display_name(cfg["hotkey"])
    hk_var = tk.StringVar(value=current_display)
    # Внутреннее значение (то что пойдёт в config)
    hk_internal = {"value": cfg["hotkey"]}

    hk_display = tk.Label(hk_frame, textvariable=hk_var, font=btn_f,
                          bg=BTN, fg=ACC, width=14, relief="flat", padx=8, pady=4)
    hk_display.pack(side="left", padx=(0, 8))

    capturing = {"active": False, "type": None}  # type: "keyboard" | "mouse"

    def stop_capture_ui():
        capturing["active"] = False
        capturing["type"] = None
        kb_btn.configure(state="normal")
        mouse_btn.configure(state="normal")
        hk_display.configure(fg=ACC)

    def apply_hotkey(internal_value, display_name):
        hk_internal["value"] = internal_value
        hk_var.set(display_name)
        stop_capture_ui()

    # Захват клавиши клавиатуры
    def start_capture_keyboard():
        if capturing["active"]:
            return
        capturing["active"] = True
        capturing["type"] = "keyboard"
        hk_var.set("[ нажмите клавишу... ]")
        hk_display.configure(fg="#f38ba8")
        kb_btn.configure(state="disabled")
        mouse_btn.configure(state="disabled")

        def do_capture():
            try:
                key = keyboard.read_hotkey(suppress=False)
                win.after(0, lambda: apply_hotkey(key.lower(), key.upper()))
            except Exception:
                win.after(0, stop_capture_ui)

        threading.Thread(target=do_capture, daemon=True).start()

    # Захват кнопки мыши
    def start_capture_mouse():
        if capturing["active"]:
            return
        capturing["active"] = True
        capturing["type"] = "mouse"
        hk_var.set("[ нажмите кнопку мыши... ]")
        hk_display.configure(fg="#f38ba8")
        kb_btn.configure(state="disabled")
        mouse_btn.configure(state="disabled")

        tmp_listener = {"ref": None}

        def on_click(x, y, button, pressed):
            if not pressed:
                return
            # Игнорируем левую кнопку (чтобы не захватить случайный клик)
            if button == pynput_mouse.Button.left:
                return
            # Определяем имя
            btn_map = {
                pynput_mouse.Button.x1:     ("mouse:x1",     "Мышь X1 (назад)"),
                pynput_mouse.Button.x2:     ("mouse:x2",     "Мышь X2 (вперёд)"),
                pynput_mouse.Button.middle: ("mouse:middle", "Мышь (колёсико)"),
                pynput_mouse.Button.right:  ("mouse:right",  "Мышь (правая)"),
            }
            internal, display = btn_map.get(button, (f"mouse:{button.name}", str(button)))
            if tmp_listener["ref"]:
                tmp_listener["ref"].stop()
            win.after(0, lambda: apply_hotkey(internal, display))

        listener = pynput_mouse.Listener(on_click=on_click)
        tmp_listener["ref"] = listener
        listener.start()

    kb_btn = tk.Button(
        hk_frame, text="Клавиатура", font=label_f,
        bg=BTN, fg=FG, activebackground=BTN_H, activeforeground=FG,
        relief="flat", padx=8, pady=3, cursor="hand2",
        command=start_capture_keyboard
    )
    kb_btn.pack(side="left", padx=(0, 4))

    mouse_btn = tk.Button(
        hk_frame, text="Мышь", font=label_f,
        bg=BTN, fg=FG, activebackground=BTN_H, activeforeground=FG,
        relief="flat", padx=8, pady=3, cursor="hand2",
        command=start_capture_mouse
    )
    mouse_btn.pack(side="left")

    # ── Подсказка по кнопкам мыши
    hint_mouse = tk.Label(win,
        text="X1/X2 — боковые кнопки мыши  •  Левую не назначать",
        font=tkfont.Font(family="Segoe UI", size=9),
        bg=BG, fg="#6c7086")
    hint_mouse.pack(anchor="w", padx=16)

    # ── Модель Whisper
    model_frame = tk.Frame(win, bg=BG)
    model_frame.pack(fill="x", **PAD)
    tk.Label(model_frame, text="Модель Whisper:", font=label_f,
             bg=BG, fg=FG, width=18, anchor="w").pack(side="left")

    model_var = tk.StringVar(value=cfg.get("model", "tiny"))
    model_combo = ttk.Combobox(model_frame, textvariable=model_var,
                               values=["tiny", "base", "small", "medium"],
                               state="readonly", width=12, font=label_f)
    model_combo.pack(side="left")
    model_hint = {"tiny": "~39 MB — быстро", "base": "~145 MB", "small": "~460 MB", "medium": "~1.5 GB"}
    hint_lbl = tk.Label(model_frame, text=model_hint.get(cfg.get("model","tiny"), ""),
                        font=label_f, bg=BG, fg="#6c7086")
    hint_lbl.pack(side="left", padx=8)
    model_var.trace_add("write", lambda *_: hint_lbl.configure(text=model_hint.get(model_var.get(), "")))

    # ── Язык
    lang_frame = tk.Frame(win, bg=BG)
    lang_frame.pack(fill="x", **PAD)
    tk.Label(lang_frame, text="Язык:", font=label_f,
             bg=BG, fg=FG, width=18, anchor="w").pack(side="left")

    lang_map = {"Русский": "ru", "Английский": "en", "Авто": "null"}
    lang_rev  = {v: k for k, v in lang_map.items()}
    lang_var  = tk.StringVar(value=lang_rev.get(cfg.get("language", "ru"), "Русский"))
    ttk.Combobox(lang_frame, textvariable=lang_var,
                 values=list(lang_map.keys()),
                 state="readonly", width=12, font=label_f).pack(side="left")

    # ── Кнопки
    tk.Frame(win, bg=ACC, height=1).pack(fill="x", padx=16, pady=(12, 0))
    btn_row = tk.Frame(win, bg=BG)
    btn_row.pack(fill="x", padx=16, pady=10)

    def on_save():
        new_cfg = {
            "hotkey":   hk_internal["value"],
            "model":    model_var.get(),
            "language": lang_map.get(lang_var.get(), "ru"),
        }
        save_config(new_cfg)
        register_hotkey(new_cfg["hotkey"])
        set_icon(ICON_IDLE, f"Voice Input: готово  [{hotkey_display_name(new_cfg['hotkey'])}]")
        if new_cfg["model"] != cfg.get("model"):
            threading.Thread(target=load_model_bg, args=(new_cfg["model"],), daemon=True).start()
        on_close()

    def on_close():
        global settings_open
        settings_open = False
        win.grab_release()
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)

    tk.Button(btn_row, text="Сохранить", font=btn_f,
              bg=ACC, fg=BG, activebackground="#b4d0ff", activeforeground=BG,
              relief="flat", padx=18, pady=5, cursor="hand2",
              command=on_save).pack(side="right", padx=(8, 0))

    tk.Button(btn_row, text="Отмена", font=label_f,
              bg=BTN, fg=FG, activebackground=BTN_H, activeforeground=FG,
              relief="flat", padx=12, pady=5, cursor="hand2",
              command=on_close).pack(side="right")


# ── Меню трея ─────────────────────────────────────────────────────────────────

def action_open_settings(icon, item):
    tk_queue.put("settings")


def action_exit(icon, item):
    tray_icon.stop()
    os._exit(0)


def build_menu():
    return pystray.Menu(
        pystray.MenuItem("Настройки", action_open_settings, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Выход", action_exit),
    )


# ── Точка входа ───────────────────────────────────────────────────────────────

import os

def main():
    global tray_icon

    cfg = load_config()

    tk_root = tk.Tk()
    tk_root.withdraw()

    def process_tk_queue():
        try:
            while True:
                task = tk_queue.get_nowait()
                if task == "settings":
                    open_settings_window(tk_root)
        except queue.Empty:
            pass
        tk_root.after(100, process_tk_queue)

    process_tk_queue()

    tray_icon = pystray.Icon(
        "VoiceInput", ICON_PROC,
        "Voice Input: запуск...",
        menu=build_menu(),
    )
    threading.Thread(target=tray_icon.run, daemon=True).start()

    threading.Thread(target=load_model_bg, args=(cfg.get("model", "tiny"),), daemon=True).start()

    register_hotkey(cfg["hotkey"])

    try:
        tk_root.mainloop()
    except KeyboardInterrupt:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
