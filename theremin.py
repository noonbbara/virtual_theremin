import cv2
import mediapipe as mp
import numpy as np
import pygame
from collections import deque
import threading
import time

# ============================================
# 1. 초기화
# ============================================
pygame.init()
# 버퍼 크기를 512로 줄여서 레이턴시 감소 (더 즉각적인 반응)
pygame.mixer.init(44100, -16, 2, 512)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ============================================
# 2. 음계 설정 (3옥타브 범위: C3 ~ C6)
# ============================================
# 영문 음계 이름 (12음계)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# 한글 음계 이름 - 영문으로 표기 (폰트 깨짐 방지)
KOR_NAMES = ['Do', 'Do#', 'Re', 'Re#', 'Mi', 'Fa', 'Fa#', 'Sol', 'Sol#', 'La', 'La#', 'Si']


def get_note_frequency(note_index):
    """
    음계 인덱스를 주파수(Hz)로 변환
    0 = C3 (130.81Hz), 36 = C6 (1046.50Hz)
    총 37개 음 (3옥타브 + 1음)
    """
    base_freq = 130.81  # C3 (낮은 도)
    return base_freq * (2 ** (note_index / 12))


def quantize_to_note(x_position):
    """
    0~1 범위의 손 x 위치를 음계 인덱스(0~36)로 변환
    """
    note_index = int(x_position * 36)
    note_index = max(0, min(36, note_index))
    return note_index


# ============================================
# 3. 연속 사운드 생성 (끊김 없는 재생)
# ============================================
# 전역 변수
current_freq = 261.63  # 현재 주파수
current_vol = 0.0  # 현재 볼륨 (0~1)
is_playing = True  # 스레드 실행 플래그
phase = 0  # 사인파 위상 (연속성 유지용)


def continuous_sound_thread():
    """
    백그라운드에서 연속적으로 사운드를 생성하는 스레드
    짧은 청크를 겹쳐서 재생하여 끊김 없는 소리 구현
    """
    global phase, current_freq, current_vol, is_playing

    sample_rate = 44100
    chunk_duration = 0.05  # 50ms 청크 (더 짧게 = 더 부드럽게)
    n_samples = int(sample_rate * chunk_duration)

    while is_playing:
        # 현재 주파수와 볼륨으로 사인파 생성
        t = np.arange(n_samples) / sample_rate

        # 위상 연속성을 유지하며 사인파 생성
        wave = np.sin(2 * np.pi * current_freq * t + phase) * current_vol * 0.3

        # 다음 청크를 위한 위상 업데이트
        phase = (phase + 2 * np.pi * current_freq * chunk_duration) % (2 * np.pi)

        # 16비트 정수로 변환 후 스테레오로 만들기
        wave_int = (wave * 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave_int, wave_int))

        try:
            # 사운드 재생
            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.play()

            # 청크가 겹치도록 대기 시간 조정 (90% 대기 = 10% 오버랩)
            time.sleep(chunk_duration * 0.85)
        except:
            pass


# 사운드 스레드 시작 (daemon=True로 메인 종료시 자동 종료)
sound_thread = threading.Thread(target=continuous_sound_thread, daemon=True)
sound_thread.start()

# ============================================
# 4. 카메라 설정
# ============================================
cap = cv2.VideoCapture(0)

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 화면 크기 조정
display_width = int(original_width * 2.5)
display_height = int(original_height * 2)

# 볼륨 스무딩을 위한 버퍼 (급격한 변화 방지)
vol_buffer = deque(maxlen=5)
current_note_index = 18  # C4부터 시작

print("=" * 50)
print("가상 테레민 - 음계 양자화 모드 (3 Octaves)")
print("=" * 50)
print("오른손 (가로): 음높이 선택 (Do~Si, 3옥타브)")
print("왼손 (세로): 볼륨 조절 (위=크게, 아래=작게)")
print("종료: 'q' 키")
print("=" * 50)

hand_detected = False
fade_out_speed = 0.05  # 손 떼었을 때 페이드아웃 속도


# ============================================
# 5. 프렛보드 시각화 함수
# ============================================
def draw_fretboard(img, current_note_index, hand_x=None):
    """
    기타 프렛처럼 음계를 시각화
    - current_note_index: 현재 선택된 음 (0~36)
    - hand_x: 손의 실제 위치 (0~1)
    """
    h, w = img.shape[:2]

    # 상단 배경 (프렛보드 영역)
    cv2.rectangle(img, (0, 0), (w, 140), (40, 40, 40), -1)

    # 37개 프렛 (3옥타브 + 1음)
    num_frets = 37
    fret_width = w / num_frets

    # 주요 음만 표시 (C, E, G만 - 3도 간격)
    major_notes = [0, 4, 7]  # C, E, G (도, 미, 솔)

    for i in range(num_frets):
        x = int(i * fret_width)

        # 현재 선택된 음 하이라이트 (초록색)
        if i == current_note_index:
            cv2.rectangle(img, (x, 0), (int(x + fret_width), 140), (0, 255, 100), -1)

        # 프렛 구분선 (얇은 회색선)
        cv2.line(img, (x, 0), (x, 140), (150, 150, 150), 1)

        note_in_octave = i % 12

        # 옥타브 시작점 (C음) 강조 - 굵은 노란선
        if note_in_octave == 0:
            cv2.line(img, (x, 0), (x, 140), (255, 255, 0), 4)

            # 옥타브 번호 표시
            octave = 3 + (i // 12)
            cv2.putText(img, f"C{octave}", (x + 5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 주요 음계만 표시 (C, E, G)
        elif note_in_octave in major_notes:
            text_x = int(x + fret_width / 2 - 10)
            octave = 3 + (i // 12)

            # 영문 음계만 표시
            cv2.putText(img, NOTE_NAMES[note_in_octave], (text_x, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 손 위치 표시 (파란 세로선)
    if hand_x is not None:
        hand_pixel_x = int(hand_x * w)
        cv2.line(img, (hand_pixel_x, 0), (hand_pixel_x, 140), (255, 100, 0), 3)
        cv2.circle(img, (hand_pixel_x, 70), 10, (255, 100, 0), -1)

    # 현재 연주 중인 음 정보 표시 (하단)
    if current_note_index < 37:
        octave = 3 + (current_note_index // 12)
        note_in_octave = current_note_index % 12
        freq = get_note_frequency(current_note_index)

        # 정보 텍스트 (한글명 + 영문명 + 주파수)
        info_text = f"Playing: {KOR_NAMES[note_in_octave]} ({NOTE_NAMES[note_in_octave]}{octave}) - {freq:.1f}Hz"
        cv2.putText(img, info_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


# ============================================
# 6. 메인 루프
# ============================================
while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("카메라를 찾을 수 없습니다.")
        break

    # 좌우 반전 (거울 모드)
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # MediaPipe로 손 감지
    results = hands_detector.process(img_rgb)

    hand_detected = False
    right_hand_x = None

    if results.multi_hand_landmarks:
        hand_detected = True

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 왼손/오른손 구분
            label = results.multi_handedness[i].classification[0].label

            # 검지 손가락 끝 (랜드마크 8번)
            index_finger = hand_landmarks.landmark[8]
            cx, cy = int(index_finger.x * w), int(index_finger.y * h)

            # 손 스켈레톤 그리기
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if label == 'Right':
                # ===== 오른손: 음계 선택 =====
                right_hand_x = index_finger.x
                current_note_index = quantize_to_note(index_finger.x)
                current_freq = get_note_frequency(current_note_index)

                # 현재 음 표시
                octave = 3 + (current_note_index // 12)
                note_in_octave = current_note_index % 12

                note_text = f"{KOR_NAMES[note_in_octave]} ({NOTE_NAMES[note_in_octave]}{octave})"
                cv2.putText(img, note_text, (cx + 15, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            else:
                # ===== 왼손: 볼륨 조절 =====
                # 위쪽 = 볼륨 크게, 아래쪽 = 볼륨 작게
                raw_vol = max(0.0, min(1.0, 1.0 - index_finger.y))
                vol_buffer.append(raw_vol)
                target_vol = np.mean(vol_buffer)

                # 볼륨 부드럽게 변화
                if current_vol < target_vol:
                    current_vol = min(current_vol + 0.15, target_vol)
                else:
                    current_vol = max(current_vol - 0.15, target_vol)

                vol_text = f"Volume: {int(current_vol * 100)}%"
                cv2.putText(img, vol_text, (cx + 15, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)

    # 손이 감지되지 않으면 볼륨 페이드아웃
    if not hand_detected:
        if current_vol > 0:
            current_vol = max(0, current_vol - fade_out_speed)

    # 프렛보드 시각화
    draw_fretboard(img, current_note_index, right_hand_x)

    # 상태 표시 (재생 중 / 무음)
    status = "PLAYING" if current_vol > 0.05 else "SILENT"
    status_color = (0, 255, 0) if current_vol > 0.05 else (100, 100, 100)
    cv2.putText(img, f"Status: {status}", (w - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # 화면 크기 1.5배로 확대
    img_resized = cv2.resize(img, (display_width, display_height))

    cv2.imshow("Virtual Theremin - Quantized Mode", img_resized)

    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================
# 7. 종료 처리
# ============================================
is_playing = False
time.sleep(0.2)  # 스레드 종료 대기
cap.release()
cv2.destroyAllWindows()
pygame.quit()
print("프로그램을 종료합니다.")