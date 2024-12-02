import cv2
import mediapipe as mp
import numpy as np
import time

# 각도 계산을 위한 함수 정의
def calculate_angle(a, b, c):
    """
    세 점 a, b, c를 받아서 각도를 계산합니다.
    a, b, c는 각각 [x, y] 형태의 리스트 또는 배열이어야 합니다.
    """
    a = np.array(a)  # 점 a
    b = np.array(b)  # 점 b (꼭지점)
    c = np.array(c)  # 점 c

    # 벡터 BA와 BC 계산
    ba = a - b
    bc = c - b

    # 벡터의 내적을 이용한 각도 계산
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)  # 라디안 단위
    angle_degrees = np.degrees(angle)  # 도 단위로 변환

    return angle_degrees

# MediaPipe 포즈 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 창 이름
window_name = '스쿼트 자세 분석'

# 창을 생성하고 기본적으로 창 모드로 설정
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

fullscreen = False  # 전체 화면 여부

while cap.isOpened():
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 프레임을 가져올 수 없습니다.")
        break

    # 이미지를 좌우 반전
    frame = cv2.flip(frame, 1)

    # BGR 이미지를 RGB로 변환
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 포즈 감지 수행
    results = pose.process(image_rgb)

    # RGB에서 다시 BGR로 변환
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # 감지된 랜드마크가 있으면
    if results.pose_landmarks:
        # 랜드마크 그리기
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 랜드마크 좌표 추출
        landmarks = results.pose_landmarks.landmark

        # 각 주요 지점의 좌표 가져오기
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # 각도 계산
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        left_hip_angle = calculate_angle(left_knee, left_hip, left_ankle)  # 고관절 각도
        right_hip_angle = calculate_angle(right_knee, right_hip, right_ankle)  # 고관절 각도
        
        left_ankle_angle = calculate_angle(left_knee, left_ankle, left_hip)  # 발목 각도
        right_ankle_angle = calculate_angle(right_knee, right_ankle, right_hip)  # 발목 각도

        # 각도 표시 위치 설정 (무릎 근처)
        image_height, image_width, _ = image.shape
        left_knee_coords = tuple(np.multiply(left_knee, [image_width, image_height]).astype(int))
        right_knee_coords = tuple(np.multiply(right_knee, [image_width, image_height]).astype(int))

        # 좌측 무릎 각도 텍스트 표시
        cv2.putText(image, f'Knee: {int(left_knee_angle)}',
                    (left_knee_coords[0] - 50, left_knee_coords[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 우측 무릎 각도 텍스트 표시
        cv2.putText(image, f'Knee: {int(right_knee_angle)}',
                    (right_knee_coords[0] + 10, right_knee_coords[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 좌측 고관절 각도 표시
        cv2.putText(image, f'Hip: {int(left_hip_angle)}',
                    (left_knee_coords[0] - 50, left_knee_coords[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 우측 고관절 각도 표시
        cv2.putText(image, f'Hip: {int(right_hip_angle)}',
                    (right_knee_coords[0] + 10, right_knee_coords[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 좌측 발목 각도 표시
        cv2.putText(image, f'Ankle: {int(left_ankle_angle)}',
                    (left_knee_coords[0] - 50, left_knee_coords[1] + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 우측 발목 각도 표시
        cv2.putText(image, f'Ankle: {int(right_ankle_angle)}',
                    (right_knee_coords[0] + 10, right_knee_coords[1] + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # FPS 계산
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    cv2.putText(image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 결과 프레임 표시
    cv2.imshow(window_name, image)

    # 'f' 키를 눌러 전체 화면 전환
    if cv2.waitKey(1) & 0xFF == ord('f'):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
