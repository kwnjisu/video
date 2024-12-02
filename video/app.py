from flask import Flask, Response, render_template, jsonify
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import json
import time
import threading
from queue import Queue
import copy

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

# 프레임 버퍼
frame_buffer = Queue(maxsize=2)
processed_frame = None
processing_lock = threading.Lock()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def process_frame_thread(body_part):
    global processed_frame
    
    while True:
        if frame_buffer.empty():
            time.sleep(0.001)  # CPU 사용률 감소
            continue
            
        frame = frame_buffer.get()
        if frame is None:
            break
            
        # BGR에서 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe Pose 처리
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # 랜드마크 그리기
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1)
            )
            
            landmarks = results.pose_landmarks.landmark
            joint_angles = {}
            
            if body_part == 'upper':
                # 상체 관절 각도 계산
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
                
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
                
                # 각도 계산
                left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, [left_shoulder[0], left_shoulder[1]-0.2])
                right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, [right_shoulder[0], right_shoulder[1]-0.2])
                
                # 어깨 각도를 180도에 가깝게 수정 (팔을 들었을 때)
                if left_shoulder_angle < 90:
                    left_shoulder_angle = 180 - left_shoulder_angle
                if right_shoulder_angle < 90:
                    right_shoulder_angle = 180 - right_shoulder_angle
                
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_wrist_angle = calculate_angle(left_elbow, left_wrist, left_index)
                right_wrist_angle = calculate_angle(right_elbow, right_wrist, right_index)
                
                # 각도 저장
                joint_angles.update({
                    'left_어깨': round(left_shoulder_angle, 1),
                    'right_어깨': round(right_shoulder_angle, 1),
                    'left_팔꿈치': round(left_elbow_angle, 1),
                    'right_팔꿈치': round(right_elbow_angle, 1),
                    'left_손목': round(left_wrist_angle, 1),
                    'right_손목': round(right_wrist_angle, 1)
                })
                
                # 각도 표시 (한 줄에 양쪽 각도)
                cv2.putText(frame, f"Shoulder: L {left_shoulder_angle:.1f} R {right_shoulder_angle:.1f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Elbow: L {left_elbow_angle:.1f} R {right_elbow_angle:.1f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Wrist: L {left_wrist_angle:.1f} R {right_wrist_angle:.1f}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            elif body_part == 'lower':
                # 하체 관절 각도 계산
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                
                # 각도 계산
                left_hip_angle = calculate_angle([left_hip[0], left_hip[1]-0.2], left_hip, left_knee)
                right_hip_angle = calculate_angle([right_hip[0], right_hip[1]-0.2], right_hip, right_knee)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                left_ankle_angle = calculate_angle(left_knee, left_ankle, left_foot_index)
                right_ankle_angle = calculate_angle(right_knee, right_ankle, right_foot_index)
                
                # 각도 저장
                joint_angles.update({
                    'left_고관절': round(left_hip_angle, 1),
                    'right_고관절': round(right_hip_angle, 1),
                    'left_무릎': round(left_knee_angle, 1),
                    'right_무릎': round(right_knee_angle, 1),
                    'left_발목': round(left_ankle_angle, 1),
                    'right_발목': round(right_ankle_angle, 1)
                })
                
                # 각도 표시 (한 줄에 양쪽 각도)
                cv2.putText(frame, f"Hip: L {left_hip_angle:.1f} R {right_hip_angle:.1f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Knee: L {left_knee_angle:.1f} R {right_knee_angle:.1f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Ankle: L {left_ankle_angle:.1f} R {right_ankle_angle:.1f}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if is_recording:
                add_joint_data(joint_angles)
        
        with processing_lock:
            processed_frame = frame.copy()

def generate_frames(body_part):
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
    
    # 처리 스레드 시작
    processing_thread = threading.Thread(target=process_frame_thread, args=(body_part,))
    processing_thread.daemon = True
    processing_thread.start()
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
                
            # 프레임 버퍼에 추가
            if not frame_buffer.full():
                frame_buffer.put(frame.copy())
            
            # 처리된 프레임 사용
            with processing_lock:
                output_frame = processed_frame if processed_frame is not None else frame
            
            # JPEG 인코딩
            ret, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    finally:
        # 정리
        frame_buffer.put(None)  # 처리 스레드 종료 신호
        processing_thread.join()
        camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select')
def select():
    return render_template('select.html')

@app.route('/upper_body')
def upper_body():
    return render_template('upper_body.html')

@app.route('/lower_body')
def lower_body():
    return render_template('lower_body.html')

@app.route('/joint_data')
def joint_data():
    try:
        # joint_data.json 파일에서 데이터 읽기
        with open('joint_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 데이터를 시간순으로 정렬
        sorted_data = sorted(data, key=lambda x: x['timestamp'])
        
        print(f"데이터 개수: {len(sorted_data)}")  # 디버깅용 출력
        
        return render_template('joint_data.html', joint_data=sorted_data)
        
    except FileNotFoundError:
        print("joint_data.json 파일을 찾을 수 없습니다.")
        return render_template('joint_data.html', joint_data=[])
        
    except Exception as e:
        print(f"데이터 로딩 중 오류 발생: {e}")
        return render_template('joint_data.html', joint_data=[])

@app.route('/video_feed_upper')
def video_feed_upper():
    return Response(generate_frames('upper'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_lower')
def video_feed_lower():
    return Response(generate_frames('lower'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_recording/<body_part>')
def start_recording(body_part):
    global is_recording, recording_start_time, current_recording_type, joint_data
    
    is_recording = True
    recording_start_time = datetime.now()
    current_recording_type = body_part
    joint_data = []  # 새운 기록 시작시 데이터 초기화
    
    return jsonify({"status": "success", "message": f"{body_part} 기록 시작"})

@app.route('/api/stop_recording/<body_part>')
def stop_recording(body_part):
    global is_recording, recording_start_time, current_recording_type
    
    is_recording = False
    recording_start_time = None
    current_recording_type = None
    
    # 데이터를 파일에 저장
    save_joint_data()
    
    return jsonify({"status": "success", "message": f"{body_part} 기록 중지"})

@app.route('/api/recording_status')
def get_recording_status():
    if not is_recording or not recording_start_time:
        return jsonify({"recording": False, "elapsed_time": "00:00"})
    
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

def add_joint_data(joint_angles):
    global joint_data
    
    if not is_recording:
        return
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 관절 쌍 정의 (손목과 발목 추가)
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
