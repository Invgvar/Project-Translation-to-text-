import os
import sys
import time
import shutil
import threading
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
try:
    import numpy as np
except Exception:
    np = None
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import PyPDF2
except Exception:
    PyPDF2 = None
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False
try:
    from pix2text import Pix2Text, merge_line_texts
    P2T_AVAILABLE = True
except Exception:
    Pix2Text = None
    merge_line_texts = None
    P2T_AVAILABLE = False
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except Exception:
    convert_from_path = None
    PDF2IMAGE_AVAILABLE = False


DEFAULT_LANG = "rus+eng"
DEFAULT_LANG_AUTO = "auto"
MODE_AUTO = "auto"
MODE_TEXT = "text"
MODE_MATH = "math"


class UILogger:
    def __init__(self, widget: ScrolledText):
        self.widget = widget
        self.lock = threading.Lock()

    def write(self, msg: str):
        with self.lock:
            self.widget.configure(state="normal")
            self.widget.insert("end", msg + "\n")
            self.widget.see("end")
            self.widget.configure(state="disabled")

    def clear(self):
        with self.lock:
            self.widget.configure(state="normal")
            self.widget.delete("1.0", "end")
            self.widget.configure(state="disabled")



def _candidate_tesseract_paths() -> list:
    paths = []
    if sys.platform.startswith("win"):
        paths += [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    else:

        paths += [
            "/opt/homebrew/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/usr/bin/tesseract",
        ]
    return paths

def ensure_tesseract(logger: Optional[UILogger] = None) -> bool:

    tpath = shutil.which("tesseract")
    if tpath:
        pytesseract.pytesseract.tesseract_cmd = tpath
        if logger:
            logger.write(f"[OK] Tesseract найден в PATH: {tpath}")
        return True


    for p in _candidate_tesseract_paths():
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            if logger:
                logger.write(f"[OK] Tesseract найден: {p}")
            return True

    if logger:
        logger.write("[WARN] Tesseract не найден. OCR может не работать.")
        logger.write("      Установи tesseract-ocr и убедись, что он в PATH.")
    return False

def detect_language_from_text(text: str) -> str:

    text = (text or "").strip()
    if not LANGDETECT_AVAILABLE or len(text) < 50:
        return DEFAULT_LANG

    try:
        code = detect(text)
    except Exception:
        return DEFAULT_LANG

    if code == "ru":
        return "rus+eng"
    if code == "en":
        return "eng"
    if code in ("el", "gr"):

        return "ell+eng"

    return DEFAULT_LANG



@dataclass
class PreprocessConfig:
    scale: int = 2
    contrast: float = 1.8
    median: int = 3
    binarize: bool = False
    bin_thresh: int = 150

def preprocess_image(image: Image.Image, cfg: PreprocessConfig) -> Image.Image:
    img = image.convert("L")

    if cfg.scale and cfg.scale != 1:
        w, h = img.size
        img = img.resize((w * cfg.scale, h * cfg.scale), Image.BICUBIC)

    if cfg.contrast and abs(cfg.contrast - 1.0) > 1e-6:
        img = ImageEnhance.Contrast(img).enhance(cfg.contrast)

    if cfg.median and cfg.median > 0:
        img = img.filter(ImageFilter.MedianFilter(size=cfg.median))

    if cfg.binarize:
        thr = int(cfg.bin_thresh)
        img = img.point(lambda p: 255 if p > thr else 0)

    return img



def tesseract_ocr(image: Image.Image, lang: str, for_math: bool) -> str:

    if for_math:
        psm = 11
        whitelist = (
            "0123456789"
            "+-−*/=()[]{}.,:;^%<>≠≈±∞πσµ·"
            "_'\"|"
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            "αβγδεζηθικλμνξοπρστυφχψω"
            "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
        )
        config = (
            f"--oem 3 --psm {psm} "
            "-c preserve_interword_spaces=1 "
            f"-c tessedit_char_whitelist={whitelist}"
        )
    else:
        psm = 6
        config = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"

    return pytesseract.image_to_string(image, lang=lang, config=config)


class Pix2TextWrapper:
    def __init__(self):
        self.model = None
        if P2T_AVAILABLE:
            try:
                self.model = Pix2Text()
            except Exception:
                self.model = None

    def available(self) -> bool:
        return self.model is not None and np is not None

    def ocr(self, image: Image.Image) -> str:

        arr = np.array(image.convert("RGB"))
        outs = self.model.recognize_text_formula(arr, resized_shape=768, return_text=True)
        try:
            return merge_line_texts(outs)
        except Exception:
            if isinstance(outs, str):
                return outs
            parts = []
            if isinstance(outs, list):
                for item in outs:
                    if isinstance(item, dict):
                        parts.append(item.get("text") or "")
                    else:
                        parts.append(str(item))
            return "\n".join([p for p in parts if p.strip()])
P2T = Pix2TextWrapper()
def extract_text_from_image_file(
    image_path: str,
    lang: str,
    mode: str,
    cfg_text: PreprocessConfig,
    cfg_math: PreprocessConfig,
    logger: Optional[UILogger] = None
) -> str:
    img = Image.open(image_path)


    if mode == MODE_MATH:
        pre = preprocess_image(img, cfg_math)
        if P2T.available():
            try:
                return P2T.ocr(pre)
            except Exception as e:
                if logger:
                    logger.write(f"[WARN] Pix2Text ошибка, fallback на Tesseract: {e}")
        return tesseract_ocr(pre, lang, for_math=True)

    if mode == MODE_TEXT:
        pre = preprocess_image(img, cfg_text)
        return tesseract_ocr(pre, lang, for_math=False)

    pre_text = preprocess_image(img, cfg_text)
    out_text = tesseract_ocr(pre_text, lang, for_math=False)
    if len((out_text or "").strip()) >= 20:
        return out_text

    pre_math = preprocess_image(img, cfg_math)
    if P2T.available():
        try:
            out_math = P2T.ocr(pre_math)
            if len((out_math or "").strip()) >= 5:
                return out_math
        except Exception as e:
            if logger:
                logger.write(f"[WARN] Pix2Text ошибка в AUTO: {e}")

    out_math2 = tesseract_ocr(pre_math, lang, for_math=True)
    return out_math2 or out_text


def extract_text_from_pdf(
    pdf_path: str,
    lang: str,
    mode: str,
    cfg_text: PreprocessConfig,
    cfg_math: PreprocessConfig,
    logger: Optional[UILogger] = None,
    cancel_flag: Optional[threading.Event] = None
) -> str:

    if pdfplumber is None:
        raise RuntimeError("Не установлен pdfplumber. Установи: pip install pdfplumber")

    out_parts = []

    with pdfplumber.open(pdf_path) as pdf:
        n = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            if cancel_flag and cancel_flag.is_set():
                if logger:
                    logger.write("[INFO] Отмена пользователем.")
                break

            page_text = (page.extract_text() or "").strip()

            need_ocr = False
            if mode == MODE_MATH:
                need_ocr = True
            elif mode == MODE_TEXT:

                need_ocr = (len(page_text) < 20)
            else:
                need_ocr = (len(page_text) < 20)

            if logger:
                logger.write(f"[PAGE {i}/{n}] selectable_text={len(page_text)} chars | OCR={'YES' if need_ocr else 'NO'}")

            if not need_ocr:
                out_parts.append(page_text)
                continue

            page_img = None

            if PDF2IMAGE_AVAILABLE:
                try:

                    images = convert_from_path(pdf_path, dpi=300, first_page=i, last_page=i)
                    if images:
                        page_img = images[0]
                except Exception as e:
                    if logger:
                        logger.write(f"[WARN] pdf2image не смог растеризовать страницу {i}: {e}")

            if page_img is None:
                try:
                    page_img = page.to_image(resolution=300).original
                except Exception as e:
                    if logger:
                        logger.write(f"[ERR] pdfplumber.to_image() не сработал на стр {i}: {e}")
                        logger.write("      Установи poppler+pdf2image (рекомендуется) или ImageMagick/Wand для pdfplumber.")
                    if page_text:
                        out_parts.append(page_text)
                    continue

            try:
                if mode in (MODE_TEXT, MODE_AUTO) and page_text:
                    ocr_txt = extract_text_from_pil(page_img, lang, MODE_TEXT, cfg_text, cfg_math, logger)
                    merged = (page_text + "\n" + ocr_txt).strip()
                    out_parts.append(merged)
                else:
                    ocr_txt = extract_text_from_pil(page_img, lang, MODE_MATH if mode == MODE_MATH else MODE_AUTO, cfg_text, cfg_math, logger)
                    out_parts.append((ocr_txt or "").strip())
            except Exception as e:
                if logger:
                    logger.write(f"[ERR] OCR ошибка на стр {i}: {e}")

    return "\n\n".join([p for p in out_parts if p.strip()])


def extract_text_from_pil(
    pil_img: Image.Image,
    lang: str,
    mode: str,
    cfg_text: PreprocessConfig,
    cfg_math: PreprocessConfig,
    logger: Optional[UILogger] = None
) -> str:

    img = pil_img

    if mode == MODE_MATH:
        pre = preprocess_image(img, cfg_math)
        if P2T.available():
            try:
                return P2T.ocr(pre)
            except Exception as e:
                if logger:
                    logger.write(f"[WARN] Pix2Text ошибка, fallback на Tesseract: {e}")
        return tesseract_ocr(pre, lang, for_math=True)

    if mode == MODE_TEXT:
        pre = preprocess_image(img, cfg_text)
        return tesseract_ocr(pre, lang, for_math=False)


    pre_text = preprocess_image(img, cfg_text)
    out_text = tesseract_ocr(pre_text, lang, for_math=False)
    if len((out_text or "").strip()) >= 20:
        return out_text

    pre_math = preprocess_image(img, cfg_math)
    if P2T.available():
        try:
            out_math = P2T.ocr(pre_math)
            if len((out_math or "").strip()) >= 5:
                return out_math
        except Exception as e:
            if logger:
                logger.write(f"[WARN] Pix2Text ошибка в AUTO: {e}")

    out_math2 = tesseract_ocr(pre_math, lang, for_math=True)
    return out_math2 or out_text


def extract_text_from_file(
    file_path: str,
    lang: str,
    mode: str,
    cfg_text: PreprocessConfig,
    cfg_math: PreprocessConfig,
    logger: Optional[UILogger] = None,
    cancel_flag: Optional[threading.Event] = None
) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path, lang, mode, cfg_text, cfg_math, logger, cancel_flag)

    if ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
        return extract_text_from_image_file(file_path, lang, mode, cfg_text, cfg_math, logger)

    raise ValueError(f"Неподдерживаемый формат: {ext}")

def create_main_window():
    root = tk.Tk()
    root.title("Преобразователь текста")
    root.geometry("980x720")


    selected_path = tk.StringVar(value="")
    lang_choice = tk.StringVar(value=DEFAULT_LANG_AUTO)
    mode_choice = tk.StringVar(value=MODE_AUTO)


    scale_text = tk.IntVar(value=2)
    scale_math = tk.IntVar(value=2)
    bin_text = tk.BooleanVar(value=False)
    bin_math = tk.BooleanVar(value=False)

    cancel_flag = threading.Event()

    frm_top = ttk.Frame(root, padding=10)
    frm_top.pack(fill="x")

    frm_mid = ttk.Frame(root, padding=10)
    frm_mid.pack(fill="x")
    frm_out = ttk.Frame(root, padding=10)
    frm_out.pack(fill="both", expand=True)
    ttk.Label(frm_top, text="Файл:").pack(side="left")
    ent = ttk.Entry(frm_top, textvariable=selected_path)
    ent.pack(side="left", fill="x", expand=True, padx=8)

    def pick_file():
        path = filedialog.askopenfilename(
            title="Выбрать файл",
            filetypes=[
                ("PDF", "*.pdf"),
                ("Images", "*.png *.jpg *.jpeg *.tiff *.bmp *.webp"),
                ("All", "*.*"),
            ],
        )
        if path:
            selected_path.set(path)

    ttk.Button(frm_top, text="Выбрать…", command=pick_file).pack(side="left")

    ttk.Label(frm_mid, text="Язык:").grid(row=0, column=0, sticky="w")
    cmb_lang = ttk.Combobox(frm_mid, textvariable=lang_choice, width=18, values=[
        DEFAULT_LANG_AUTO,
        "rus+eng",
        "eng",
        "ell+eng",
        "rus",
    ])
    cmb_lang.grid(row=0, column=1, padx=6, sticky="w")

    ttk.Label(frm_mid, text="Режим:").grid(row=0, column=2, sticky="w")
    cmb_mode = ttk.Combobox(frm_mid, textvariable=mode_choice, width=10, values=[
        MODE_AUTO, MODE_TEXT, MODE_MATH
    ])
    cmb_mode.grid(row=0, column=3, padx=6, sticky="w")

    ttk.Label(frm_mid, text="Scale(текст):").grid(row=0, column=4, sticky="w")
    ttk.Spinbox(frm_mid, from_=1, to=4, textvariable=scale_text, width=5).grid(row=0, column=5, padx=6, sticky="w")

    ttk.Label(frm_mid, text="Scale(формулы):").grid(row=0, column=6, sticky="w")
    ttk.Spinbox(frm_mid, from_=1, to=4, textvariable=scale_math, width=5).grid(row=0, column=7, padx=6, sticky="w")

    ttk.Checkbutton(frm_mid, text="Бинаризация(текст)", variable=bin_text).grid(row=1, column=1, sticky="w", pady=6)
    ttk.Checkbutton(frm_mid, text="Бинаризация(формулы)", variable=bin_math).grid(row=1, column=3, sticky="w", pady=6)

    ttk.Label(frm_out, text="Результат:").pack(anchor="w")
    txt_out = ScrolledText(frm_out, height=18, wrap="word")
    txt_out.pack(fill="both", expand=True)
    txt_out.insert("end", "")
    txt_out.configure(state="disabled")

    ttk.Label(frm_out, text="Журнал:").pack(anchor="w", pady=(10, 0))
    txt_log = ScrolledText(frm_out, height=10, wrap="word")
    txt_log.pack(fill="x")
    txt_log.configure(state="disabled")
    logger = UILogger(txt_log)

    frm_btn = ttk.Frame(root, padding=10)
    frm_btn.pack(fill="x")

    btn_run = ttk.Button(frm_btn, text="Распознать")
    btn_cancel = ttk.Button(frm_btn, text="Отмена")
    btn_copy = ttk.Button(frm_btn, text="Копировать результат")
    btn_clear = ttk.Button(frm_btn, text="Очистить")

    btn_run.pack(side="left")
    btn_cancel.pack(side="left", padx=8)
    btn_copy.pack(side="left")
    btn_clear.pack(side="left", padx=8)

    prog = ttk.Progressbar(frm_btn, mode="indeterminate")
    prog.pack(side="right", fill="x", expand=True, padx=8)

    def set_output(text: str):
        txt_out.configure(state="normal")
        txt_out.delete("1.0", "end")
        txt_out.insert("end", text)
        txt_out.configure(state="disabled")

    def on_copy():
        root.clipboard_clear()
        root.clipboard_append(txt_out.get("1.0", "end-1c"))
        messagebox.showinfo("Ок", "Скопировано в буфер обмена.")

    def on_clear():
        set_output("")
        logger.clear()

    btn_copy.configure(command=on_copy)
    btn_clear.configure(command=on_clear)

    def run_ocr():
        cancel_flag.clear()
        logger.clear()

        path = selected_path.get().strip()
        if not path:
            messagebox.showwarning("Нет файла", "Выбери файл.")
            return

        ensure_tesseract(logger)
        cfg_t = PreprocessConfig(
            scale=int(scale_text.get()),
            contrast=1.8,
            median=3,
            binarize=bool(bin_text.get()),
            bin_thresh=150,
        )
        cfg_m = PreprocessConfig(
            scale=int(scale_math.get()),
            contrast=2.0,
            median=3,
            binarize=bool(bin_math.get()),
            bin_thresh=150,
        )

        mode = mode_choice.get()
        lang = lang_choice.get()

        def job():
            start = time.time()
            try:
                if lang == DEFAULT_LANG_AUTO:
                    detected = DEFAULT_LANG
                    try:
                        ext = os.path.splitext(path)[1].lower()
                        if ext == ".pdf" and pdfplumber is not None:
                            sample = ""
                            with pdfplumber.open(path) as pdf:
                                for pg in pdf.pages[:3]:
                                    sample += (pg.extract_text() or "") + "\n"
                                    if len(sample) > 800:
                                        break
                            detected = detect_language_from_text(sample)
                        else:
                            img = Image.open(path)
                            pre = preprocess_image(img, cfg_t)
                            quick = tesseract_ocr(pre, DEFAULT_LANG, for_math=False)
                            detected = detect_language_from_text(quick)
                    except Exception:
                        detected = DEFAULT_LANG

                    lang_used = detected
                else:
                    lang_used = lang

                logger.write(f"[INFO] Файл: {path}")
                logger.write(f"[INFO] Язык: {lang_used}")
                logger.write(f"[INFO] Режим: {mode}")
                logger.write(f"[INFO] pdf2image: {'YES' if PDF2IMAGE_AVAILABLE else 'NO'} | pix2text: {'YES' if P2T.available() else 'NO'}")

                text = extract_text_from_file(
                    path, lang_used, mode, cfg_t, cfg_m, logger, cancel_flag
                )

                dur = time.time() - start
                logger.write(f"[DONE] Время: {dur:.2f} сек")

                root.after(0, lambda: set_output(text))

            except Exception as e:
                dur = time.time() - start
                err = f"{e}\n\n{traceback.format_exc()}"
                root.after(0, lambda: set_output(""))
                root.after(0, lambda: logger.write(f"[FAIL] {dur:.2f} сек\n{err}"))

            finally:
                root.after(0, lambda: (prog.stop(), btn_run.configure(state="normal"), btn_cancel.configure(state="disabled")))

        btn_run.configure(state="disabled")
        btn_cancel.configure(state="normal")
        prog.start(10)

        th = threading.Thread(target=job, daemon=True)
        th.start()

    def on_cancel():
        cancel_flag.set()
        logger.write("[INFO] Попросили отмену…")

    btn_run.configure(command=run_ocr)
    btn_cancel.configure(command=on_cancel, state="disabled")
    logger.write("[INFO] Готово. Выбери PDF/картинку и нажми «Распознать».")
    logger.write("[INFO] Совет: для формул чаще лучше режим 'math' + pix2text (если установлен).")

    return root

if __name__ == "__main__":
    app = create_main_window()
    app.mainloop()
