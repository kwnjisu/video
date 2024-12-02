from flask import Flask, Response, render_template, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import json
import time
import threading
from queue import Queue
import copy
import os
import absl.logging

# 로그 초기화
absl.logging.set_verbosity(absl.logging.INFO)

app = Flask(__name__)

# MediaPipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    model_complexity=0,
    smooth_landmarks=True,
    enable_segmentation=False
)

# 전역 변수
joint_data = []
recording_start_time = None
is_recording = False
current_recording_type = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upper_body')
def upper_body():
    return render_template('upper_body.html')

@app.route('/lower_body')
def lower_body():
    return render_template('lower_body.html')

@app.route('/select')
def select():
    return render_template('select.html')

@app.route('/joint_data')
def joint_data_page():
    return render_template('joint_data.html')

@app.route('/api/start_recording/<body_part>')
def start_recording(body_part):
    global recording_start_time, is_recording, current_recording_type, joint_data
    
    recording_start_time = datetime.now()
    is_recording = True
    current_recording_type = body_part
    joint_data = []
    
    return jsonify({"status": "success"})

@app.route('/api/stop_recording/<body_part>')
def stop_recording(body_part):
    global recording_start_time, is_recording, current_recording_type
    
    recording_start_time = None
    is_recording = False
    current_recording_type = None
    
    save_joint_data()
    
    return jsonify({"status": "success"})

@app.route('/api/recording_status')
def recording_status():
    if not is_recording or recording_start_time is None:
        return jsonify({"recording": False})
    
    elapsed = datetime.now() - recording_start_time
    minutes = int(elapsed.total_seconds() // 60)
    seconds = int(elapsed.total_seconds() % 60)
    
    return jsonify({
        "recording": True,
        "elapsed_time": f"{minutes:02d}:{seconds:02d}"
    })

@app.route('/api/joint_data')
def get_joint_data():
    return jsonify(joint_data)

@app.route('/api/update_joint_data', methods=['POST'])
def update_joint_data():
    if not is_recording:
        return jsonify({"status": "not recording"})
        
    data = request.json
    angles = data['angles']
    timestamp = data['timestamp']
    
    # 관절 데이터 저장
    add_joint_data(angles)
    
    return jsonify({"status": "success"})

def add_joint_data(joint_angles):
    global joint_data
    
    if not is_recording:
        return
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 관절 쌍 정의
    joint_pairs = {
        '어깨': ('left_어깨', 'right_어깨'),
        '팔꿈치': ('left_팔꿈치', 'right_팔꿈치'),
        '손목': ('left_손목', 'right_손목'),
        '고관절': ('left_고관절', 'right_고관절'),
        '무릎': ('left_무릎', 'right_무릎'),
        '발목': ('left_발목', 'right_발목')
    }
    
    # 각 관절 쌍에 대해 하나의 데이터 항목 생성
    for joint_name, (left_key, right_key) in joint_pairs.items():
        if left_key in joint_angles or right_key in joint_angles:
            joint_data.append({
                "timestamp": timestamp,
                "joint": joint_name,
                "left_angle": joint_angles.get(left_key, 0),
                "right_angle": joint_angles.get(right_key, 0)
            })

def save_joint_data():
    """데이터를 JSON 파일로 저장"""
    try:
        with open('joint_data.json', 'w', encoding='utf-8') as f:
            json.dump(joint_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"데이터 저장 중 오류 발생: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
