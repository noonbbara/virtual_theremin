import cv2
import mediapipe as mp
import numpy as np
import pygame
from collections import deque

# ============================================
# 1. 초기화
# ============================================
pygame.init()
pygame.mixer.init(44100, -16, 2, 512)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ============================================
# 2. 음계 설정 (3옥타브: C3 ~ C6)
# ============================================
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
KOR_NAMES  = ['Do','Do#','Re','Re#','Mi','Fa','Fa#','Sol','Sol#','La','La#','Si']

def get_note_frequency(note_index):
    base_freq = 130.81  # C3
    return base_freq * (2 ** (note_index / 12.0))

def quantize_to_note(x_position):
    note_index = int(x_position * 36)
    return max(0, min(36, note_index))

# ============================================
# 3. 음계별 루프 사운드 미리 생성
# ============================================
SAMPLE_RATE  = 44100
LOOP_DURATION = 2.0
LOOP_SAMPLES  = int(SAMPLE_RATE * LOOP_DURATION)

def create_note_sound(note_index):
    freq = get_note_frequency(note_index)
    t = np.linspace(0, LOOP_DURATION, LOOP_SAMPLES, endpoint=False)
    wave = np.sin(2 * np.pi * freq * t) * 0.3
    wave_int = (wave * 32767).astype(np.int16)
    stereo  = np.column_stack((wave_int, wave_int))
    return pygame.sndarray.make_sound(stereo)

print("음계 사운드 미리 생성 중...")
NOTE_SOUNDS = { i: create_note_sound(i) for i in range(37) }
print("완료!")

playing_channel      = None
current_playing_note = -1

# ============================================
# 4. 왼손 주먹/펄침 감지
# ============================================
def is_fist(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    closed = sum(1 for tip, pip in zip(finger_tips, finger_pips)
                 if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y)
    return closed >= 3

# ============================================
# 5. 재생 제어
# ============================================
def play_note(note_index):
    global playing_channel, current_playing_note
    if note_index == current_playing_note and playing_channel is not None:
        return
    if playing_channel is not None:
        playing_channel.stop()
    playing_channel = NOTE_SOUNDS[note_index].play(-1)
    playing_channel.set_volume(current_vol)
    current_playing_note = note_index

def stop_note():
    global playing_channel, current_playing_note
    if playing_channel is not None:
        playing_channel.stop()
        playing_channel = None
        current_playing_note = -1

# ============================================
# 6. 카메라 설정
# ============================================
cap = cv2.VideoCapture(0)

original_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
display_width   = int(original_width * 2.5)
display_height  = int(original_height * 2)

vol_buffer          = deque(maxlen=5)
current_note_index  = 18
current_vol         = 0.0
is_muted            = True

print("=" * 50)
print("가상 테레민 - 양자화 모드 (3 Octaves)")
print("=" * 50)
print("오른손 (가로): 음높이 선택")
print("왼손 펴진 상태 (세로): 볼륨 조절")
print("왼손 주먹쥬기: 즉시 뮤트")
print("종료: 'q' 키")
print("=" * 50)

# ============================================
# 7. 시각화 — 깔끔한 바 + 기준점만
# ============================================
def draw_ui(img, note_index, hand_x=None, volume=0.0, muted=False):
    h, w = img.shape[:2]

    BAR_Y     = 60   # 음계 바 수직 중앙
    BAR_H     = 8    # 바 두께
    BAR_LEFT  = 40
    BAR_RIGHT = w - 40
    BAR_W     = BAR_RIGHT - BAR_LEFT

    # ---- 배경 바 (어두운 회색) ----
    cv2.rectangle(img, (BAR_LEFT, BAR_Y - BAR_H//2),
                       (BAR_RIGHT, BAR_Y + BAR_H//2), (60, 60, 60), -1)

    # ---- 옥타브 기준점 (C3, C4, C5, C6) ----
    octave_indices = [0, 12, 24, 36]  # C3, C4, C5, C6
    for idx in octave_indices:
        px = int(BAR_LEFT + (idx / 36.0) * BAR_W)
        # 작은 세로선
        cv2.line(img, (px, BAR_Y - 14), (px, BAR_Y + 14), (180, 180, 180), 2)
        # 옥타브 라벨
        octave = 3 + (idx // 12)
        cv2.putText(img, f"C{octave}", (px - 8, BAR_Y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # ---- 현재 선택된 음계 포인터 (밝은 초록 원) ----
    note_px = int(BAR_LEFT + (note_index / 36.0) * BAR_W)
    cv2.circle(img, (note_px, BAR_Y), 10, (0, 255, 100), -1)
    cv2.circle(img, (note_px, BAR_Y), 10, (0, 200, 80), 2)  # 테두리

    # ---- 손 실제 위치 포인터 (주황 작은 원) ----
    if hand_x is not None:
        hand_px = int(BAR_LEFT + hand_x * BAR_W)
        cv2.circle(img, (hand_px, BAR_Y), 4, (255, 140, 0), -1)

    # ---- 현재 음 라벨 (포인터 아래) ----
    note_in_octave = note_index % 12
    octave = 3 + (note_index // 12)
    freq   = get_note_frequency(note_index)
    label  = f"{KOR_NAMES[note_in_octave]} ({NOTE_NAMES[note_in_octave]}{octave}) {freq:.0f}Hz"
    # 텍스트를 포인터 중앙 아래에 배치
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_x = max(BAR_LEFT, min(note_px - tw // 2, BAR_RIGHT - tw))
    cv2.putText(img, label, (text_x, BAR_Y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)

    # ---- 볼륨 바 (왼쪽 세로) ----
    VOL_X      = 18
    VOL_TOP    = 30
    VOL_BOTTOM = h - 60
    VOL_H      = VOL_BOTTOM - VOL_TOP

    # 배경
    cv2.rectangle(img, (VOL_X - 4, VOL_TOP), (VOL_X + 4, VOL_BOTTOM), (60, 60, 60), -1)
    # 채운 부분 (아래서 위로)
    fill_y = int(VOL_BOTTOM - volume * VOL_H)
    if not muted and volume > 0:
        cv2.rectangle(img, (VOL_X - 4, fill_y), (VOL_X + 4, VOL_BOTTOM), (0, 180, 255), -1)
    # 볼륨 포인터 원
    cv2.circle(img, (VOL_X, fill_y), 6, (0, 200, 255), -1)

    # ---- 뮤트 표시 ----
    if muted:
        cv2.rectangle(img, (0, h - 36), (w, h), (0, 0, 160), -1)
        cv2.putText(img, "MUTED — Open left hand to play", (16, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # ---- 상태 (우측 상단) ----
    if muted:
        status, sc = "MUTED",   (0, 0, 255)
    elif volume > 0.05:
        status, sc = "PLAYING", (0, 255, 0)
    else:
        status, sc = "SILENT",  (100, 100, 100)
    cv2.putText(img, status, (w - 100, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)

# ============================================
# 8. 메인 루프
# ============================================
while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("카메라를 찾을 수 없습니다.")
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands_detector.process(img_rgb)

    right_hand_x    = None
    left_hand_found = False

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label

            index_finger = hand_landmarks.landmark[8]
            cx, cy = int(index_finger.x * w), int(index_finger.y * h)

            # 검지 끝점만 하나 표시
            cv2.circle(img, (cx, cy), 8, (255, 255, 255), -1)
            cv2.circle(img, (cx, cy), 8, (0, 0, 0), 2)

            if label == 'Right':
                right_hand_x       = index_finger.x
                current_note_index = quantize_to_note(index_finger.x)

                if not is_muted:
                    play_note(current_note_index)

            else:
                left_hand_found = True

                if is_fist(hand_landmarks):
                    if not is_muted:
                        is_muted = True
                        stop_note()
                        current_vol = 0.0
                else:
                    was_muted = is_muted
                    is_muted  = False

                    raw_vol = max(0.0, min(1.0, 1.0 - index_finger.y))
                    vol_buffer.append(raw_vol)
                    current_vol = float(np.mean(vol_buffer))

                    if playing_channel is not None:
                        playing_channel.set_volume(current_vol)

                    if was_muted:
                        play_note(current_note_index)

    if not results.multi_hand_landmarks:
        if not is_muted:
            is_muted = True
            stop_note()
            current_vol = 0.0

    # UI 그리기
    draw_ui(img, current_note_index, hand_x=right_hand_x,
            volume=current_vol, muted=is_muted)

    img_resized = cv2.resize(img, (display_width, display_height))
    cv2.imshow("Virtual Theremin", img_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================
# 9. 종료 처리
# ============================================
stop_note()
cap.release()
cv2.destroyAllWindows()
pygame.quit()
print("프로그램을 종료합니다.")