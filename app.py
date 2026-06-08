import os
import threading
from flask import Flask, render_template, request, send_file, jsonify

from analyzer import analyze_video

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}

task_status = {
    "running": False,
    "done": False,
    "error": None,
    "percent": 0,
    "message": "대기 중",
    "report": None,
    "summary": None,
}


def allowed_file(filename):
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def update_progress(percent, message):
    task_status["percent"] = percent
    task_status["message"] = message


def run_analysis(input_path, output_csv_path, output_report_path):
    global task_status

    try:
        task_status["running"] = True
        task_status["done"] = False
        task_status["error"] = None
        task_status["percent"] = 0
        task_status["message"] = "분석 준비 중..."
        task_status["report"] = None
        task_status["summary"] = None

        result = analyze_video(
            input_path,
            output_csv_path,
            output_report_path,
            progress_callback=update_progress
        )

        task_status["report"] = result["report"]
        task_status["summary"] = result["summary"]
        task_status["percent"] = 100
        task_status["message"] = "분석 완료"
        task_status["done"] = True

    except Exception as e:
        task_status["error"] = str(e)
        task_status["message"] = "오류 발생"
        task_status["done"] = True

    finally:
        task_status["running"] = False


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    print("Analyze request received", flush=True)
    if task_status["running"]:
        return jsonify({
            "ok": False,
            "error": "이미 분석이 진행 중입니다."
        })

    if "video" not in request.files:
        return jsonify({
            "ok": False,
            "error": "영상 파일이 업로드되지 않았습니다."
        })

    file = request.files["video"]

    if file.filename == "":
        return jsonify({
            "ok": False,
            "error": "선택된 파일이 없습니다."
        })

    if not allowed_file(file.filename):
        return jsonify({
            "ok": False,
            "error": "mp4, mov, avi, mkv 파일만 업로드할 수 있습니다."
        })

    input_path = os.path.join(UPLOAD_FOLDER, "input_video.mp4")
    output_csv_path = os.path.join(OUTPUT_FOLDER, "result.csv")
    output_report_path = os.path.join(OUTPUT_FOLDER, "report.txt")

    file.save(input_path)
    print("File saved:", input_path, flush=True)

    thread = threading.Thread(
        target=run_analysis,
        args=(input_path, output_csv_path, output_report_path)
    )
    thread.start()
    print("Analysis thread started", flush=True)

    return jsonify({
        "ok": True,
        "message": "분석을 시작했습니다."
    })


@app.route("/progress")
def progress():
    return jsonify(task_status)


@app.route("/download/csv")
def download_csv():
    return send_file(
        os.path.join(OUTPUT_FOLDER, "result.csv"),
        as_attachment=True
    )


@app.route("/download/report")
def download_report():
    return send_file(
        os.path.join(OUTPUT_FOLDER, "report.txt"),
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(debug=True)