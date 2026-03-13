import serial
import time

SERIAL_PORT = '/dev/ttyACM0'
BAUDRATE = 1000000
SERVO_ID = 6

def read_reg(ser, sid, reg, size):
    msg = [sid, 0x04, 0x02, reg, size]
    checksum = ~(sum(msg)) & 0xFF
    packet = bytes([0xFF, 0xFF] + msg + [checksum])
    ser.reset_input_buffer()
    ser.write(packet)
    time.sleep(0.02)
    if ser.in_waiting >= (6 + size):
        res = ser.read(ser.in_waiting)
        try:
            idx = res.index(b'\xff\xff')
            return int.from_bytes(res[idx+5 : idx+5+size], byteorder='little')
        except: return None
    return None

def write_reg(ser, sid, reg, size, value):
    val_bytes = value.to_bytes(size, byteorder='little')
    msg = [sid, size + 3, 0x03, reg] + list(val_bytes)
    checksum = ~(sum(msg)) & 0xFF
    packet = bytes([0xFF, 0xFF] + msg + [checksum])
    ser.write(packet)
    time.sleep(0.05)

def interrogate():
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.5)
    print(f"--- 正在审讯 ID={SERVO_ID} 的底层参数 ---")
    
    # 1. 读取固件设定的硬性限位
    min_pos = read_reg(ser, SERVO_ID, 9, 2)
    max_pos = read_reg(ser, SERVO_ID, 11, 2)
    cur_pos = read_reg(ser, SERVO_ID, 56, 2)
    speed = read_reg(ser, SERVO_ID, 46, 2)
    
    print(f"🛑 内部物理限位 -> 最小: {min_pos}, 最大: {max_pos}")
    print(f"📍 当前实际位置 -> {cur_pos}")
    print(f"⚡ 当前运行速度 -> {speed} (如果为0，可能导致不转)")

    if cur_pos is None or min_pos is None:
        print("❌ 读取失败，请重试。")
        ser.close(); return

    # 2. 智能计算安全目标位置
    # 如果当前位置靠近最大值，我们就往回退 100；如果靠近最小值，就往前走 100
    if cur_pos + 100 > max_pos:
        safe_target = cur_pos - 150
        print(f"⚠️ 刚才的2243确实越界了！自动计算安全退让目标: {safe_target}")
    else:
        safe_target = cur_pos + 150
        print(f"✅ 位置空间充足，计算安全前进目标: {safe_target}")

    # 3. 强行注入速度并测试移动
    print("\n--- 开始安全移动测试 ---")
    write_reg(ser, SERVO_ID, 40, 1, 1)        # 确保扭矩开启
    write_reg(ser, SERVO_ID, 46, 2, 1000)     # 强行设置安全运行速度为 1000
    write_reg(ser, SERVO_ID, 42, 2, safe_target) # 发送安全目标位置
    
    print(f"👉 已下发新位置: {safe_target}，等待 0.5 秒...")
    time.sleep(0.5)
    
    # 4. 验证是否真的动了
    new_pos = read_reg(ser, SERVO_ID, 56, 2)
    print(f"📍 执行后当前位置 -> {new_pos}")
    
    if abs(new_pos - safe_target) < 20:
        print("🎉 破案了！电机正常转动。之前不转是因为你撞到了固件的安全限位（或者速度被清零了）。")
    elif abs(new_pos - cur_pos) < 10:
        print("❌ 见鬼了，在限位内且有速度的情况下依然不转。")
        load = read_reg(ser, SERVO_ID, 60, 2)
        print(f"💪 此时负载为: {load}。如果大于100，说明物理结构彻底卡死了（检查螺丝和轴承！）。")
    
    ser.close()

if __name__ == "__main__":
    interrogate()