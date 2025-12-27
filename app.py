from flask import Flask, render_template, redirect, url_for, request, flash
import subprocess
import sys
import os
import csv
from datetime import date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
attendance_process = None


app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

app.secret_key = "attendance-secret"

attendance_process = None  # prevent multiple attendance runs



@app.route("/")
def dashboard():
    # ---- Total students ----
    students = 0
    faces_dir = os.path.join(BASE_DIR, "known_faces")
    if os.path.exists(faces_dir):
        students = len([
            d for d in os.listdir(faces_dir)
            if os.path.isdir(os.path.join(faces_dir, d))
        ])

    # ---- Attendance stats ----
    total_attendance = 0
    today_attendance = 0
    recent_records = []

    today = date.today().isoformat()
    csv_path = os.path.join(BASE_DIR, "attendance.csv")

    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None) 
            rows = list(reader)

            total_attendance = len(rows)
            today_attendance = sum(1 for r in rows if r[1] == today)
            recent_records = rows[-5:][::-1]

    return render_template(
        "dashboard.html",
        students=students,
        total_attendance=total_attendance,
        today_attendance=today_attendance,
        recent_records=recent_records
    )



@app.route("/add_student", methods=["POST"])
def add_student():
    name = request.form.get("student_name")

    if not name:
        flash("Student name is required!", "error")
        return redirect(url_for("dashboard"))

   
    subprocess.run([
        sys.executable,
        os.path.join(BASE_DIR, "collect_faces.py"),
        name
    ])


    subprocess.run([
        sys.executable,
        os.path.join(BASE_DIR, "train_faces.py")
    ])

    flash(f"Student '{name}' registered successfully 📸", "success")
    return redirect(url_for("dashboard"))



@app.route("/attendance")
def attendance():
    global attendance_process

    if attendance_process is None or attendance_process.poll() is not None:
        attendance_process = subprocess.Popen(
            [sys.executable, os.path.join(BASE_DIR, "attendance.py")]
        )
        flash("Attendance started 📷 (Camera opened)", "success")
    else:
        flash("Attendance is already running", "error")

    return redirect(url_for("dashboard"))

@app.route("/stop_attendance")
def stop_attendance():
    global attendance_process

    if attendance_process and attendance_process.poll() is None:
        attendance_process.terminate()
        attendance_process = None
        flash("Attendance stopped 🛑", "success")
    else:
        flash("Attendance is not running", "error")

    return redirect(url_for("dashboard"))



@app.route("/records")
def records():
    records = []
    csv_path = os.path.join(BASE_DIR, "attendance.csv")

    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for name, date_, time_ in reader:
                records.append({
                    "name": name,
                    "date": date_,
                    "time": time_
                })

    return render_template("records.html", records=records)



if __name__ == "__main__":
    app.run(debug=True)
