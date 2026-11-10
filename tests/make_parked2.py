import cv2, numpy as np, random
random.seed(5); cap = cv2.VideoCapture("clip.mp4"); frames = {}
for n in range(1, 340):
    ok, f = cap.read()
    if not ok: break
    if 318 <= n <= 336: frames[n] = f
out = cv2.VideoWriter("parked2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (704, 576)); blank = np.full((576, 704, 3), 90, np.uint8)
for n in range(318, 337): out.write(frames[n])
base = frames[336]; lost = 0
for i in range(25 * 240):                                                        # 4 minutes parked
    if (i % 1125) < 125 or random.random() < 0.12: out.write(blank); lost += 1; continue      # 5 s blackout every 45 s + 12% single blank frames
    dx, dy = random.uniform(-6, 6), random.uniform(-6, 6); f = cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, dy]]), (704, 576), borderMode=cv2.BORDER_REPLICATE); out.write(f)
out.release(); print("blank/lost frames:", lost, "of", 25 * 240)
