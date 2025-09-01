#!/usr/bin/env python3
import time
import argparse
import cv2
from flask import Flask, Response

app = Flask(__name__)

def mjpeg_generator(video_path: str, fps: float | None, loop: bool, quality: int, width: int | None, height: int | None):
    # open per-client so each connection is independent
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    out_fps = float(fps) if fps and fps > 0 else (float(src_fps) if src_fps > 0 else 30.0)
    period = 1.0 / out_fps
    next_due = time.perf_counter()

    try:
        while True:
            now = time.perf_counter()
            if now < next_due:
                # tight sleep to hit the cadence
                time.sleep(min(0.001, next_due - now))
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            if width or height:
                h, w = frame.shape[:2]
                if width and height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                elif width:
                    new_h = int(h * (width / float(w)))
                    frame = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
                elif height:
                    new_w = int(w * (height / float(h)))
                    frame = cv2.resize(frame, (new_w, height), interpolation=cv2.INTER_AREA)

            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if not ok:
                continue
            jpg = buf.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                + jpg
                + b"\r\n"
            )

            next_due += period
            # if we fell behind badly, jump the schedule forward
            if (time.perf_counter() - next_due) > (2 * period):
                next_due = time.perf_counter() + period
    finally:
        cap.release()

@app.route("/stream.mjpg")
def stream():
    gen = mjpeg_generator(app.config["VIDEO"], app.config["FPS"], app.config["LOOP"],
                          app.config["QUALITY"], app.config["WIDTH"], app.config["HEIGHT"])
    return Response(gen, mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    p = argparse.ArgumentParser(description="MJPEG streamer for an MP4 (real-time).")
    p.add_argument("--video", default="demo.mp4", help="Path to MP4")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--fps", type=float, default=0.0, help="Override FPS (0=use source or 30)")
    p.add_argument("--loop", action="store_true", help="Loop at EOF")
    p.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100")
    p.add_argument("--width", type=int, default=0, help="Optional resize width")
    p.add_argument("--height", type=int, default=0, help="Optional resize height")
    args = p.parse_args()

    app.config.update(
        VIDEO=args.video,
        FPS=args.fps,
        LOOP=args.loop,
        QUALITY=max(1, min(100, args.quality)),
        WIDTH=(args.width or None),
        HEIGHT=(args.height or None),
    )
    print(f"Serving {args.video} at http://{args.host}:{args.port}/stream.mjpg")
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    main()
