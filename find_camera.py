import cv2

def check_cameras():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Camera Index {i}", frame)
                print(f"索引 {i} 可用，请观察弹出的窗口是否为你想要的画面。")
                cv2.waitKey(2000) # 显示2秒
            cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_cameras()