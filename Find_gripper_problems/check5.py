import serial
import time
import sys

# --- 配置区 ---
SERIAL_PORT = 'COM5'  # 请确认设备管理器中目前的 COM 口
BAUDRATE = 1000000
SERVO_IDS = range(1, 7) # SO-ARM101 的 6 个关节

def read_reg(ser, sid, reg, size):
    # 构造飞特读取指令包 (FF FF ID Length 0x02 Reg Size Checksum)
    msg = [sid, 0x04, 0x02, reg, size]
    checksum = ~(sum(msg)) & 0xFF
    packet = bytes([0xFF, 0xFF] + msg + [checksum])
    
    ser.reset_input_buffer()
    ser.write(packet)
    time.sleep(0.02) # 给硬件一点处理和回传的时间
    
    expected_len = 6 + size
    if ser.in_waiting >= expected_len:
        res = ser.read(ser.in_waiting)
        try:
            idx = res.index(b'\xff\xff')
            data = res[idx+5 : idx+5+size]
            return int.from_bytes(data, byteorder='little')
        except ValueError:
            return None
    return None

def check_all_servos():
    print(f"=== 🚀 开始 SO-ARM101 底层六轴全检 ===")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.2, write_timeout=0.2)
        print(f"✅ 成功连接端口: {SERIAL_PORT}\n")
    except Exception as e:
        print(f"❌ 无法打开串口，请确保 LeRobot 等其他程序已关闭: {e}")
        sys.exit()

    offline_servos = []
    locked_servos = []

    for sid in SERVO_IDS:
        print(f"▶️ 检测 ID = {sid} ...", end=" ")
        
        # 1. 查当前位置 (测试是否在线)
        cur_pos = read_reg(ser, sid, 56, 2)
        
        if cur_pos is None:
            print("❌ [无响应/离线]")
            offline_servos.append(sid)
            continue
            
        # 2. 查限位和硬件错误状态
        min_pos = read_reg(ser, sid, 9, 2)
        max_pos = read_reg(ser, sid, 11, 2)
        hw_error = read_reg(ser, sid, 41, 1) # 硬件错误状态 (0为正常)
        
        print(f"✅ 在线 | 位置: {cur_pos:4d} | 限位: [{min_pos}, {max_pos}] | 错误码: {hw_error}")
        
        # 3. 诊断物理锁死问题
        if min_pos is not None and max_pos is not None:
            # 如果最大值和最小值相等，或者区间极小（比如小于10），说明被锁死了
            if abs(max_pos - min_pos) <= 5:
                locked_servos.append(sid)

    ser.close()

    print("\n=== 📊 诊断报告 ===")
    if not offline_servos and not locked_servos:
        print("🎉 完美！所有 6 个舵机均在线，且未被限位锁死。你可以放心地去跑 LeRobot 评估了。")
    else:
        if offline_servos:
            print(f"⚠️ 离线舵机 ID: {offline_servos} (请顺着线检查，可能是接头松了)")
        if locked_servos:
            print(f"🛑 被限位锁死的舵机 ID: {locked_servos} (Min 和 Max 几乎相等，电机拒绝转动)")

if __name__ == "__main__":
    check_all_servos()