from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    module: str
    command: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _run_id(module: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = module.lower().replace(" ", "_")
    return f"{stamp}_{safe}"


class TrainingGUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Training GUI")
        self.resize(1100, 800)

        self._process: QProcess | None = None
        self._stdout_path: Path | None = None
        self._stderr_path: Path | None = None
        self._run_context: RunContext | None = None
        self._config_dir = _repo_root() / "config" / "gui"

        self._tabs = QTabWidget()
        self._clip_tab = self._build_clip_tab()
        self._labeler_tab = self._build_labeler_tab()
        self._planner_tab = self._build_placeholder_tab("Planner training is not implemented.")
        self._controller_tab = self._build_placeholder_tab("Controller training is not implemented.")
        self._tabs.addTab(self._clip_tab, "Clip Extractor")
        self._tabs.addTab(self._labeler_tab, "VLM Labeler")
        self._tabs.addTab(self._planner_tab, "Planner Training")
        self._tabs.addTab(self._controller_tab, "Controller Training")

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setLineWrapMode(QPlainTextEdit.NoWrap)

        self._start_btn = QPushButton("Start")
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._clear_log_btn = QPushButton("Clear Log")
        self._status_label = QLabel("Idle")
        self._run_label = QLabel("-")

        self._start_btn.clicked.connect(self._start_clicked)
        self._stop_btn.clicked.connect(self._stop_clicked)
        self._clear_log_btn.clicked.connect(self._log_view.clear)

        controls = QHBoxLayout()
        controls.addWidget(self._start_btn)
        controls.addWidget(self._stop_btn)
        controls.addWidget(self._clear_log_btn)
        controls.addStretch(1)
        controls.addWidget(QLabel("Run:"))
        controls.addWidget(self._run_label)
        controls.addWidget(QLabel("Status:"))
        controls.addWidget(self._status_label)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._tabs)
        main_layout.addLayout(controls)
        main_layout.addWidget(self._log_view, 1)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        self._ensure_config_dir()
        self._load_or_init_configs()

    def _build_clip_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()

        self._clip_zip = self._path_row("sessions.zip", file_mode=True)
        self._clip_output = self._path_row("output dir", dir_mode=True)
        self._clip_fps = QSpinBox()
        self._clip_fps.setRange(1, 60)
        self._clip_fps.setValue(2)
        self._clip_step = QSpinBox()
        self._clip_step.setRange(1, 1000)
        self._clip_step.setValue(2)
        self._clip_allow_partial = QCheckBox("allow partial windows")
        self._clip_export = QCheckBox("export clips")
        self._clip_export_ratio = QDoubleSpinBox()
        self._clip_export_ratio.setRange(0.0, 1.0)
        self._clip_export_ratio.setSingleStep(0.01)
        self._clip_export_ratio.setValue(0.01)
        self._clip_link_mode = QComboBox()
        self._clip_link_mode.addItems(["hardlink", "symlink", "copy"])
        self._clip_seed = QSpinBox()
        self._clip_seed.setRange(0, 1_000_000)
        self._clip_seed.setValue(0)

        form = QFormLayout()
        form.addRow("zip path", self._clip_zip)
        form.addRow("output dir", self._clip_output)
        form.addRow("fps", self._clip_fps)
        form.addRow("step", self._clip_step)
        form.addRow("", self._clip_allow_partial)
        form.addRow("", self._clip_export)
        form.addRow("export ratio", self._clip_export_ratio)
        form.addRow("link mode", self._clip_link_mode)
        form.addRow("seed", self._clip_seed)

        group = QGroupBox("Clip Extractor")
        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch(1)
        tab.setLayout(layout)
        return tab

    def _build_labeler_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()

        self._label_input_dir = self._path_row("input dir", dir_mode=True)
        self._label_backend = QComboBox()
        self._label_backend.addItems(["openai", "ollama"])
        self._label_base_url = QLineEdit()
        self._label_model = QLineEdit()
        self._label_api_key = QLineEdit()
        self._label_api_key.setEchoMode(QLineEdit.Password)
        self._label_batch = QSpinBox()
        self._label_batch.setRange(1, 128)
        self._label_batch.setValue(8)
        self._label_retries = QSpinBox()
        self._label_retries.setRange(0, 10)
        self._label_retries.setValue(3)
        self._label_timeout = QDoubleSpinBox()
        self._label_timeout.setRange(1.0, 600.0)
        self._label_timeout.setValue(120.0)
        self._label_temperature = QDoubleSpinBox()
        self._label_temperature.setRange(0.0, 2.0)
        self._label_temperature.setSingleStep(0.1)
        self._label_temperature.setValue(0.0)
        self._label_limit = QSpinBox()
        self._label_limit.setRange(0, 1_000_000)
        self._label_limit.setValue(0)
        self._label_include_enums = QCheckBox("include enums")
        self._label_include_enums.setChecked(True)
        self._label_validate = QCheckBox("validate output")
        self._label_validate.setChecked(True)
        self._label_flush = QCheckBox("flush every batch")
        self._label_flush.setChecked(True)
        self._label_ollama_format = QLineEdit("json")
        self._label_ollama_num_predict = QSpinBox()
        self._label_ollama_num_predict.setRange(0, 4096)
        self._label_ollama_num_predict.setValue(0)

        form = QFormLayout()
        form.addRow("input dir", self._label_input_dir)
        form.addRow("backend", self._label_backend)
        form.addRow("base url", self._label_base_url)
        form.addRow("model", self._label_model)
        form.addRow("api key", self._label_api_key)
        form.addRow("batch size", self._label_batch)
        form.addRow("max retries", self._label_retries)
        form.addRow("timeout sec", self._label_timeout)
        form.addRow("temperature", self._label_temperature)
        form.addRow("limit (0=all)", self._label_limit)
        form.addRow("", self._label_include_enums)
        form.addRow("", self._label_validate)
        form.addRow("", self._label_flush)
        form.addRow("ollama format", self._label_ollama_format)
        form.addRow("ollama num_predict", self._label_ollama_num_predict)

        group = QGroupBox("VLM Labeler")
        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch(1)
        tab.setLayout(layout)
        return tab

    def _build_placeholder_tab(self, text: str) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        tab.setLayout(layout)
        return tab

    def _path_row(self, label: str, file_mode: bool = False, dir_mode: bool = False) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        button = QPushButton("Browse")

        def pick_path() -> None:
            if file_mode:
                path, _ = QFileDialog.getOpenFileName(self, f"Select {label}")
            elif dir_mode:
                path = QFileDialog.getExistingDirectory(self, f"Select {label}")
            else:
                path, _ = QFileDialog.getOpenFileName(self, f"Select {label}")
            if path:
                edit.setText(path)

        button.clicked.connect(pick_path)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        widget.setLayout(layout)
        widget._edit = edit  # type: ignore[attr-defined]
        return widget

    def _get_path_value(self, widget: QWidget) -> str:
        return widget._edit.text().strip()  # type: ignore[attr-defined]

    def _ensure_config_dir(self) -> None:
        self._config_dir.mkdir(parents=True, exist_ok=True)

    def _config_path(self, module: str) -> Path:
        return self._config_dir / f"{module}.json"

    def _load_module_config(self, module: str) -> dict[str, Any]:
        path = self._config_path(module)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save_module_config(self, module: str, data: dict[str, Any]) -> None:
        path = self._config_path(module)
        path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")

    def _load_or_init_configs(self) -> None:
        clip = self._load_module_config("clip_extractor")
        if clip:
            self._apply_clip_config(clip)
        else:
            self._save_module_config("clip_extractor", self._collect_clip_config())

        labeler = self._load_module_config("vlm_labeler")
        if labeler:
            self._apply_labeler_config(labeler)
        else:
            self._save_module_config("vlm_labeler", self._collect_labeler_config())

        if not self._config_path("planner").exists():
            self._save_module_config("planner", {"module": "planner"})
        if not self._config_path("controller").exists():
            self._save_module_config("controller", {"module": "controller"})

    def _collect_clip_config(self) -> dict[str, Any]:
        return {
            "zip_path": self._get_path_value(self._clip_zip),
            "output_dir": self._get_path_value(self._clip_output),
            "fps": self._clip_fps.value(),
            "step": self._clip_step.value(),
            "allow_partial": self._clip_allow_partial.isChecked(),
            "export_clips": self._clip_export.isChecked(),
            "export_ratio": self._clip_export_ratio.value(),
            "link_mode": self._clip_link_mode.currentText(),
            "seed": self._clip_seed.value(),
        }

    def _apply_clip_config(self, config: dict[str, Any]) -> None:
        if "zip_path" in config:
            self._clip_zip._edit.setText(str(config["zip_path"]))  # type: ignore[attr-defined]
        if "output_dir" in config:
            self._clip_output._edit.setText(str(config["output_dir"]))  # type: ignore[attr-defined]
        if "fps" in config:
            self._clip_fps.setValue(int(config["fps"]))
        if "step" in config:
            self._clip_step.setValue(int(config["step"]))
        if "allow_partial" in config:
            self._clip_allow_partial.setChecked(bool(config["allow_partial"]))
        if "export_clips" in config:
            self._clip_export.setChecked(bool(config["export_clips"]))
        if "export_ratio" in config:
            self._clip_export_ratio.setValue(float(config["export_ratio"]))
        if "link_mode" in config:
            idx = self._clip_link_mode.findText(str(config["link_mode"]))
            if idx >= 0:
                self._clip_link_mode.setCurrentIndex(idx)
        if "seed" in config:
            self._clip_seed.setValue(int(config["seed"]))

    def _collect_labeler_config(self) -> dict[str, Any]:
        return {
            "input_dir": self._get_path_value(self._label_input_dir),
            "backend": self._label_backend.currentText(),
            "base_url": self._label_base_url.text().strip(),
            "model": self._label_model.text().strip(),
            "api_key": self._label_api_key.text(),
            "batch_size": self._label_batch.value(),
            "max_retries": self._label_retries.value(),
            "timeout_sec": self._label_timeout.value(),
            "temperature": self._label_temperature.value(),
            "limit": self._label_limit.value(),
            "include_enums": self._label_include_enums.isChecked(),
            "validate": self._label_validate.isChecked(),
            "flush_every_batch": self._label_flush.isChecked(),
            "ollama_format": self._label_ollama_format.text().strip(),
            "ollama_num_predict": self._label_ollama_num_predict.value(),
        }

    def _apply_labeler_config(self, config: dict[str, Any]) -> None:
        if "input_dir" in config:
            self._label_input_dir._edit.setText(str(config["input_dir"]))  # type: ignore[attr-defined]
        if "backend" in config:
            idx = self._label_backend.findText(str(config["backend"]))
            if idx >= 0:
                self._label_backend.setCurrentIndex(idx)
        if "base_url" in config:
            self._label_base_url.setText(str(config["base_url"]))
        if "model" in config:
            self._label_model.setText(str(config["model"]))
        if "api_key" in config:
            self._label_api_key.setText(str(config["api_key"]))
        if "batch_size" in config:
            self._label_batch.setValue(int(config["batch_size"]))
        if "max_retries" in config:
            self._label_retries.setValue(int(config["max_retries"]))
        if "timeout_sec" in config:
            self._label_timeout.setValue(float(config["timeout_sec"]))
        if "temperature" in config:
            self._label_temperature.setValue(float(config["temperature"]))
        if "limit" in config:
            self._label_limit.setValue(int(config["limit"]))
        if "include_enums" in config:
            self._label_include_enums.setChecked(bool(config["include_enums"]))
        if "validate" in config:
            self._label_validate.setChecked(bool(config["validate"]))
        if "flush_every_batch" in config:
            self._label_flush.setChecked(bool(config["flush_every_batch"]))
        if "ollama_format" in config:
            self._label_ollama_format.setText(str(config["ollama_format"]))
        if "ollama_num_predict" in config:
            self._label_ollama_num_predict.setValue(int(config["ollama_num_predict"]))

    def _start_clicked(self) -> None:
        if self._process is not None:
            QMessageBox.warning(self, "Busy", "A task is already running.")
            return

        current = self._tabs.currentWidget()
        if current is self._planner_tab or current is self._controller_tab:
            QMessageBox.information(self, "Not Implemented", "This module is not implemented yet.")
            return

        if current is self._clip_tab:
            module, command, config = self._build_clip_command()
        else:
            module, command, config = self._build_labeler_command()

        if command is None:
            return

        run_id = _run_id(module)
        run_dir = _repo_root() / "runs" / run_id
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._stdout_path = log_dir / "stdout.log"
        self._stderr_path = log_dir / "stderr.log"
        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=True, indent=2), encoding="utf-8")

        self._run_context = RunContext(run_id=run_id, run_dir=run_dir, module=module, command=command)
        self._run_label.setText(run_id)
        self._status_label.setText("Running")
        self._log_view.clear()

        if module == "clip_extractor":
            self._save_module_config("clip_extractor", self._collect_clip_config())
        if module == "vlm_labeler":
            self._save_module_config("vlm_labeler", self._collect_labeler_config())

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.readyReadStandardOutput.connect(self._handle_stdout)
        process.readyReadStandardError.connect(self._handle_stderr)
        process.finished.connect(self._process_finished)

        self._process = process
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        program = command[0]
        args = command[1:]
        process.start(program, args)

    def _stop_clicked(self) -> None:
        if not self._process:
            return
        self._append_log("[info] terminating process")
        self._process.terminate()
        QTimer.singleShot(3000, self._force_kill_if_running)

    def _force_kill_if_running(self) -> None:
        if self._process and self._process.state() != QProcess.NotRunning:
            self._append_log("[warn] force killing process")
            self._process.kill()

    def _process_finished(self) -> None:
        self._append_log("[info] process finished")
        self._process = None
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_label.setText("Idle")

    def _handle_stdout(self) -> None:
        if not self._process:
            return
        data = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self._append_log(data, stream="stdout")

    def _handle_stderr(self) -> None:
        if not self._process:
            return
        data = self._process.readAllStandardError().data().decode("utf-8", errors="replace")
        self._append_log(data, stream="stderr")

    def _append_log(self, text: str, stream: str | None = None) -> None:
        if not text:
            return
        if stream:
            prefix = f"[{stream}] "
        else:
            prefix = ""
        for line in text.splitlines():
            entry = prefix + line
            self._log_view.appendPlainText(entry)
            log_path = self._stdout_path
            if stream == "stderr":
                log_path = self._stderr_path or self._stdout_path
            if log_path:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(entry + "\n")

    def _build_clip_command(self) -> tuple[str, list[str] | None, dict[str, Any]]:
        zip_path = self._get_path_value(self._clip_zip)
        out_dir = self._get_path_value(self._clip_output)
        if not zip_path or not out_dir:
            QMessageBox.warning(self, "Missing input", "zip path and output dir are required.")
            return "clip_extractor", None, {}

        script = _repo_root() / "scripts" / "clip_extractor.py"
        cmd = [
            sys.executable,
            str(script),
            "--zip",
            zip_path,
            "--output",
            out_dir,
            "--fps",
            str(self._clip_fps.value()),
            "--step",
            str(self._clip_step.value()),
            "--export-ratio",
            str(self._clip_export_ratio.value()),
            "--link-mode",
            self._clip_link_mode.currentText(),
            "--seed",
            str(self._clip_seed.value()),
        ]
        if self._clip_allow_partial.isChecked():
            cmd.append("--allow-partial")
        if self._clip_export.isChecked():
            cmd.append("--export-clips")

        config = {
            "module": "clip_extractor",
            "zip_path": zip_path,
            "output_dir": out_dir,
            "fps": self._clip_fps.value(),
            "step": self._clip_step.value(),
            "allow_partial": self._clip_allow_partial.isChecked(),
            "export_clips": self._clip_export.isChecked(),
            "export_ratio": self._clip_export_ratio.value(),
            "link_mode": self._clip_link_mode.currentText(),
            "seed": self._clip_seed.value(),
            "command": cmd,
        }
        return "clip_extractor", cmd, config

    def _build_labeler_command(self) -> tuple[str, list[str] | None, dict[str, Any]]:
        input_dir = self._get_path_value(self._label_input_dir)
        base_url = self._label_base_url.text().strip()
        model = self._label_model.text().strip()
        api_key = self._label_api_key.text().strip()
        backend = self._label_backend.currentText()
        if not input_dir or not base_url or not model:
            QMessageBox.warning(self, "Missing input", "input dir, base url, model are required.")
            return "vlm_labeler", None, {}

        script = _repo_root() / "scripts" / "vlm_labeler.py"
        cmd = [
            sys.executable,
            str(script),
            "--input-dir",
            input_dir,
            "--base-url",
            base_url,
            "--model",
            model,
            "--backend",
            backend,
            "--batch-size",
            str(self._label_batch.value()),
            "--max-retries",
            str(self._label_retries.value()),
            "--timeout",
            str(self._label_timeout.value()),
            "--temperature",
            str(self._label_temperature.value()),
        ]
        if api_key:
            cmd.extend(["--api-key", api_key])
        if not self._label_include_enums.isChecked():
            cmd.append("--no-enums")
        if not self._label_validate.isChecked():
            cmd.append("--skip-validation")
        if not self._label_flush.isChecked():
            cmd.append("--no-flush-every-batch")
        if self._label_limit.value() > 0:
            cmd.extend(["--limit", str(self._label_limit.value())])

        if backend == "ollama":
            format_value = self._label_ollama_format.text().strip()
            if format_value:
                cmd.extend(["--ollama-format", format_value])
            if self._label_ollama_num_predict.value() > 0:
                cmd.extend(["--ollama-num-predict", str(self._label_ollama_num_predict.value())])

        config = {
            "module": "vlm_labeler",
            "input_dir": input_dir,
            "base_url": base_url,
            "model": model,
            "backend": backend,
            "batch_size": self._label_batch.value(),
            "max_retries": self._label_retries.value(),
            "timeout_sec": self._label_timeout.value(),
            "temperature": self._label_temperature.value(),
            "limit": self._label_limit.value(),
            "include_enums": self._label_include_enums.isChecked(),
            "validate": self._label_validate.isChecked(),
            "flush_every_batch": self._label_flush.isChecked(),
            "ollama_format": self._label_ollama_format.text().strip(),
            "ollama_num_predict": self._label_ollama_num_predict.value(),
            "command": cmd,
        }
        return "vlm_labeler", cmd, config

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_module_config("clip_extractor", self._collect_clip_config())
        self._save_module_config("vlm_labeler", self._collect_labeler_config())
        if not self._config_path("planner").exists():
            self._save_module_config("planner", {"module": "planner"})
        if not self._config_path("controller").exists():
            self._save_module_config("controller", {"module": "controller"})
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = TrainingGUI()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
