import cv2

# 测试你刚才填的两个序号：1 和 2
cap_workspace = cv2.VideoCapture(1)
cap_hand = cv2.VideoCapture(2)

# 尝试分别抓取一帧
ret1, frame1 = cap_workspace.read()
ret2, frame2 = cap_hand.read()

print(f"--- 摄像头测试结果 ---")
print(f"Workspace (序号 1) 抓取状态: {ret1}")
if ret1:
    print(f"   -> 画面大小: {frame1.shape}")

print(f"\nHand (序号 2) 抓取状态: {ret2}")
if ret2:
    print(f"   -> 画面大小: {frame2.shape}")

cap_workspace.release()
cap_hand.release()