import os
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
import customtkinter as ctk

# =====================================================================
# 1. STRICT GPU ENFORCEMENT (NO CPU FALLBACK)
# Must be applied BEFORE basic_pitch imports onnxruntime
# =====================================================================
import onnxruntime as ort

AVAILABLE_PROVIDERS = ort.get_available_providers()
GPU_PROVIDERS = [
    p for p in ["DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider"]
    if p in AVAILABLE_PROVIDERS
]

ACTIVE_GPU_PROVIDER = GPU_PROVIDERS[0] if GPU_PROVIDERS else None
_OriginalInferenceSession = ort.InferenceSession


class StrictGPUInferenceSession(_OriginalInferenceSession):
    """
    Wraps ort.InferenceSession to forbid CPUExecutionProvider fallback.
    If the GPU cannot execute the model, it raises a hard runtime error.
    """
    def __init__(self, path_or_bytes, sess_options=None, providers=None, **kwargs):
        if not ACTIVE_GPU_PROVIDER:
            raise RuntimeError(
                f"FATAL: No GPU Execution Provider found! "
                f"Available providers: {AVAILABLE_PROVIDERS}. "
                f"Ensure 'onnxruntime-directml' (AMD/Intel/NVIDIA) or 'onnxruntime-gpu' is installed."
            )

        if sess_options is None:
            sess_options = ort.SessionOptions()

        # Strictly forbid ONNX Runtime from dropping nodes back to the CPU
        sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

        # Override whatever providers basic-pitch requested with ONLY the GPU provider
        super().__init__(
            path_or_bytes,
            sess_options=sess_options,
            providers=[ACTIVE_GPU_PROVIDER],
            **kwargs
        )
        # Disable runtime fallback at the session level
        self.disable_fallback()


# Monkey-patch ONNX Runtime globally before importing basic_pitch
ort.InferenceSession = StrictGPUInferenceSession

from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH


def get_onnx_model_path() -> Path:
    """Locates the bundled nmp.onnx file inside the environment or PyInstaller bundle."""
    base_path = Path(ICASSP_2022_MODEL_PATH)
    onnx_candidate = base_path.parent / "nmp.onnx"
    if onnx_candidate.exists():
        return onnx_candidate
    if base_path.suffix == ".onnx" and base_path.exists():
        return base_path
    # Fallback search inside PyInstaller _MEIPASS if needed
    if hasattr(sys, "_MEIPASS"):
        meipass_onnx = Path(sys._MEIPASS) / "basic_pitch" / "saved_models" / "icassp_2022" / "nmp.onnx"
        if meipass_onnx.exists():
            return meipass_onnx
    raise FileNotFoundError(f"Could not locate Basic Pitch ONNX model at {onnx_candidate}")


# =====================================================================
# 2. FUTURISTIC GUI APPLICATION
# =====================================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class CyberPitchApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("NEURAL // PITCH - GPU Audio-to-MIDI Extractor")
        self.geometry("680x500")
        self.resizable(False, False)
        self.configure(fg_color="#07090E")

        # State variables
        self.input_audio_path: Path | None = None
        self.midi_data = None
        self.model: Model | None = None

        self._build_ui()
        self._verify_gpu_status()

    def _build_ui(self):
        # Outer glowing-style frame
        self.main_card = ctk.CTkFrame(
            self,
            fg_color="#0D111A",
            corner_radius=16,
            border_width=2,
            border_color="#00F0FF"
        )
        self.main_card.pack(fill="both", expand=True, padx=24, pady=24)

        # Header
        self.title_lbl = ctk.CTkLabel(
            self.main_card,
            text="NEURAL // PITCH",
            font=ctk.CTkFont(family="Consolas", size=30, weight="bold"),
            text_color="#00F0FF"
        )
        self.title_lbl.pack(pady=(22, 2))

        self.subtitle_lbl = ctk.CTkLabel(
            self.main_card,
            text="POLYPHONIC AI MIDI TELEMETRY  |  ZERO-CPU FALLBACK MODE",
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color="#6C7A9C"
        )
        self.subtitle_lbl.pack(pady=(0, 14))

        # GPU Hardware Status Pill
        self.gpu_badge = ctk.CTkLabel(
            self.main_card,
            text="SCANNING HARDWARE...",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            fg_color="#141B2D",
            corner_radius=8,
            padx=14,
            pady=6
        )
        self.gpu_badge.pack(pady=(0, 18))

        # File Display Box
        self.file_box = ctk.CTkEntry(
            self.main_card,
            placeholder_text="NO AUDIO STREAM SELECTED...",
            font=ctk.CTkFont(family="Consolas", size=12),
            height=38,
            fg_color="#07090E",
            border_color="#1E2942",
            text_color="#E2E8F0",
            state="readonly"
        )
        self.file_box.pack(fill="x", padx=36, pady=(0, 16))

        # 1. Button: Choose Input Audio File
        self.btn_select = ctk.CTkButton(
            self.main_card,
            text="[ 01 ]  SELECT INPUT AUDIO FILE",
            font=ctk.CTkFont(family="Consolas", size=14, weight="bold"),
            height=46,
            corner_radius=8,
            fg_color="#112233",
            hover_color="#1A3650",
            border_width=1,
            border_color="#00F0FF",
            text_color="#00F0FF",
            command=self.select_input_file
        )
        self.btn_select.pack(fill="x", padx=36, pady=6)

        # 2. Button: Start AI Detection
        self.btn_detect = ctk.CTkButton(
            self.main_card,
            text="[ 02 ]  INITIALIZE GPU MIDI EXTRACTION",
            font=ctk.CTkFont(family="Consolas", size=14, weight="bold"),
            height=46,
            corner_radius=8,
            fg_color="#2A0E46",
            hover_color="#41166B",
            border_width=1,
            border_color="#BD00FF",
            text_color="#E6B8FF",
            state="disabled",
            command=self.start_detection
        )
        self.btn_detect.pack(fill="x", padx=36, pady=6)

        # 3. Button: Save MIDI to Directory
        self.btn_save = ctk.CTkButton(
            self.main_card,
            text="[ 03 ]  EXPORT MIDI TO DIRECTORY",
            font=ctk.CTkFont(family="Consolas", size=14, weight="bold"),
            height=46,
            corner_radius=8,
            fg_color="#0A2E26",
            hover_color="#124D3F",
            border_width=1,
            border_color="#00FF9D",
            text_color="#00FF9D",
            state="disabled",
            command=self.save_midi_to_directory
        )
        self.btn_save.pack(fill="x", padx=36, pady=6)

        # Futuristic Progress Bar
        self.progress = ctk.CTkProgressBar(
            self.main_card,
            height=8,
            corner_radius=4,
            fg_color="#141B2D",
            progress_color="#00F0FF"
        )
        self.progress.pack(fill="x", padx=36, pady=(20, 8))
        self.progress.set(0)

        # Telemetry / Status Console Line
        self.status_lbl = ctk.CTkLabel(
            self.main_card,
            text="SYS_STATUS // STANDBY",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color="#8A99AD"
        )
        self.status_lbl.pack(pady=(4, 10))

    def _verify_gpu_status(self):
        if ACTIVE_GPU_PROVIDER == "DmlExecutionProvider":
            self.gpu_badge.configure(
                text="GPU LOCKED // DIRECTML (NVIDIA / AMD / INTEL DX12)",
                text_color="#00FF9D"
            )
        elif ACTIVE_GPU_PROVIDER == "CUDAExecutionProvider":
            self.gpu_badge.configure(
                text="GPU LOCKED // NVIDIA CUDA",
                text_color="#00FF9D"
            )
        elif ACTIVE_GPU_PROVIDER == "ROCMExecutionProvider":
            self.gpu_badge.configure(
                text="GPU LOCKED // AMD ROCm",
                text_color="#00FF9D"
            )
        else:
            self.gpu_badge.configure(
                text="ERROR // NO COMPATIBLE GPU PROVIDER DETECTED",
                text_color="#FF3366"
            )
            self.status_lbl.configure(
                text="SYS_ERROR // CPU FALLBACK IS FORBIDDEN. EXECUTION LOCKED.",
                text_color="#FF3366"
            )
            self.btn_select.configure(state="disabled")

    def select_input_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("All Files", "*.*")
            ]
        )
        if not file_path:
            return

        self.input_audio_path = Path(file_path)
        self.midi_data = None
        self.btn_save.configure(state="disabled")

        self.file_box.configure(state="normal")
        self.file_box.delete(0, "end")
        self.file_box.insert(0, str(self.input_audio_path))
        self.file_box.configure(state="readonly")

        self.btn_detect.configure(state="normal")
        self.progress.set(0)
        self.status_lbl.configure(
            text=f"SYS_READY // LOADED '{self.input_audio_path.name}'",
            text_color="#00F0FF"
        )

    def start_detection(self):
        if not self.input_audio_path or not self.input_audio_path.exists():
            messagebox.showerror("Input Error", "Selected audio file does not exist.")
            return

        # Lock UI controls during inference
        self.btn_select.configure(state="disabled")
        self.btn_detect.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        self.progress.configure(mode="indeterminate", progress_color="#BD00FF")
        self.progress.start()
        self.status_lbl.configure(
            text=f"EXECUTING // RUNNING NEURAL INFERENCE ON {ACTIVE_GPU_PROVIDER}...",
            text_color="#E6B8FF"
        )

        threading.Thread(target=self._inference_worker, daemon=True).start()

    def _inference_worker(self):
        try:
            onnx_path = get_onnx_model_path()
            if self.model is None:
                self.model = Model(onnx_path)

            # Run Basic Pitch inference using our GPU-locked ONNX session
            _, midi_data, note_events = predict(
                audio_path=self.input_audio_path,
                model_or_model_path=self.model
            )

            self.midi_data = midi_data
            note_count = len(note_events)
            self.after(0, self._on_inference_complete, True, f"EXTRACTED {note_count} MIDI NOTES")
        except Exception as exc:
            self.after(0, self._on_inference_complete, False, str(exc))

    def _on_inference_complete(self, success: bool, detail: str):
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.btn_select.configure(state="normal")
        self.btn_detect.configure(state="normal")

        if success:
            self.progress.configure(progress_color="#00FF9D")
            self.progress.set(1.0)
            self.btn_save.configure(state="normal")
            self.status_lbl.configure(
                text=f"SYS_SUCCESS // {detail} — READY FOR EXPORT",
                text_color="#00FF9D"
            )
        else:
            self.progress.configure(progress_color="#FF3366")
            self.progress.set(0)
            self.status_lbl.configure(
                text="SYS_FAULT // GPU INFERENCE FAILED",
                text_color="#FF3366"
            )
            messagebox.showerror("GPU Inference Error", f"Inference aborted:\n\n{detail}")

    def save_midi_to_directory(self):
        if self.midi_data is None or self.input_audio_path is None:
            return

        target_dir = filedialog.askdirectory(title="Select Destination Folder for MIDI")
        if not target_dir:
            return

        output_filename = f"{self.input_audio_path.stem}_basic_pitch.mid"
        output_path = Path(target_dir) / output_filename

        try:
            self.midi_data.write(str(output_path))
            self.status_lbl.configure(
                text=f"SAVED // {output_path}",
                text_color="#00FF9D"
            )
            messagebox.showinfo(
                "Export Complete",
                f"MIDI file successfully written to:\n{output_path}"
            )
        except Exception as exc:
            messagebox.showerror("File Save Error", f"Could not write MIDI file:\n{exc}")


if __name__ == "__main__":
    app = CyberPitchApp()
    app.mainloop()
