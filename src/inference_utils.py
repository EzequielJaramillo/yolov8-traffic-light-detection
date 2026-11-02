"""Utilidades de entrenamiento e inferencia sobre video con Ultralytics."""
import argparse
import subprocess
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

DEFAULT_WEIGHTS = "runs/detect/runs/traffic_light/weights/best.pt"
DEFAULT_VIDEO = "data/inference_traffic_light_video.mp4"
DEFAULT_OUTPUT = "assets/inference_output.mp4"


def convert_to_h264(video_path):
    """Convierte un MP4 a formato H.264 compatible con navegadores y VS Code."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        video_path = Path(video_path)
        tmp_path = video_path.parent / f"{video_path.stem}_h264_tmp.mp4"
        cmd = [
            ffmpeg, "-y", "-i", str(video_path),
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "23", str(tmp_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and tmp_path.exists():
            tmp_path.replace(video_path)
    except Exception as e:
        print(f"Aviso H.264: {e}")


def train_model(data_yaml, model_name="yolov8n.pt", epochs=3, imgsz=320,
                batch=16, device="cpu", project="runs", name="traffic_light", patience=50):
    """Entrena YOLO y devuelve (modelo, ruta a best.pt)."""
    import os
    workers = 0 if (device == "cpu" or os.name == "nt") else 2
    model = YOLO(model_name)
    results = model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz,
                          batch=batch, device=device, project=project,
                          name=name, exist_ok=True, workers=workers, plots=True,
                          patience=patience)
    best = Path(results.save_dir) / "weights" / "best.pt"
    return model, best


def process_video(weights, video_in, video_out, imgsz=320, conf=0.25, device="cpu",
                  show=False, window_name="Deteccion de Semaforos - Inference Video (q para salir)"):
    """Ejecuta detección frame a frame y guarda un mp4 anotado.

    ``show`` controla la vista en vivo mientras se ejecutan las predicciones:
      - ``"notebook"``: reproduce el video clasificado DENTRO de la celda de Jupyter
        actualizando los frames continuamente con detecciones y métricas.
      - ``True`` / ``"window"``: abre una ventana interactiva de OpenCV en el escritorio
        mostrando el video con las clasificaciones en vivo.
      - ``False``: procesamiento silencioso sin visualización en pantalla.
    """
    mode = "notebook" if show == "notebook" else ("window" if show else None)
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {video_in}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    video_out = Path(video_out)
    video_out.parent.mkdir(parents=True, exist_ok=True)
    raw_tmp_out = video_out.parent / f"{video_out.stem}_raw.mp4"
    writer = cv2.VideoWriter(str(raw_tmp_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if mode == "window":
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, min(w, 960), min(h, 540))
        except cv2.error as e:
            print(f"Ventana no disponible ({e}); se procesa sin ventana gráfica.")
            mode = None
    elif mode == "notebook":
        from IPython.display import clear_output, display as ipy_display, Image as IPyImage

    n = 0
    t_start = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.time()
            res = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
            annotated = res.plot()
            writer.write(annotated)
            n += 1

            if mode:
                view = annotated.copy()
                fps_instant = 1 / max(time.time() - t0, 1e-6)
                label = f"Frame {n}/{total} | {fps_instant:.1f} FPS" if total else f"Frame {n} | {fps_instant:.1f} FPS"
                cv2.putText(view, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

                if mode == "window":
                    try:
                        cv2.imshow(window_name, view)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            print("\nVentana cerrada por el usuario ('q'); finalizando guardado...")
                            mode = None
                            cv2.destroyAllWindows()
                    except cv2.error:
                        mode = None
                elif mode == "notebook":
                    _, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    clear_output(wait=True)
                    ipy_display(IPyImage(data=buf.tobytes()))

            if mode != "notebook":
                pct = f" ({100 * n / total:.0f}%)" if total else ""
                print(f"\rProcesando y clasificando video: frame {n}/{total}{pct}", end="", flush=True)
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n{n} frames procesados en {elapsed:.1f} s ({n / max(elapsed, 1e-6):.1f} FPS).")
    print(f"Codificando video web H.264 en {video_out}...")
    if raw_tmp_out.exists():
        raw_tmp_out.replace(video_out)
        convert_to_h264(video_out)
    print(f"Video guardado y listo en: {video_out}")
    return video_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Reproduce y clasifica el video de inferencia en tiempo real en una ventana.")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Ruta a los pesos del modelo .pt")
    ap.add_argument("--video", default=DEFAULT_VIDEO, help="Ruta al video de prueba")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Ruta donde guardar el video resultante")
    ap.add_argument("--imgsz", type=int, default=320, help="Tamaño de entrada para inferencia")
    ap.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza")
    ap.add_argument("--device", default="cpu", help="Dispositivo ('cpu' o '0' para GPU)")
    ap.add_argument("--no-show", action="store_true", help="No abrir la ventana interactiva")
    a = ap.parse_args()

    print("=" * 70)
    print("DETECCIÓN DE SEMÁFOROS EN TIEMPO REAL - INFERENCE VIDEO")
    print("=" * 70)
    print(f"Pesos:  {a.weights}")
    print(f"Video:  {a.video}")
    print(f"Salida: {a.output}")
    print(f"Dispositivo: {a.device} | imgsz: {a.imgsz} | conf: {a.conf}")
    if not a.no_show:
        print("\nAbriendo ventana en vivo con el video clasificado (presiona 'q' para salir)...")
    print("=" * 70)

    process_video(
        weights=a.weights,
        video_in=a.video,
        video_out=a.output,
        imgsz=a.imgsz,
        conf=a.conf,
        device=a.device,
        show=not a.no_show
    )
