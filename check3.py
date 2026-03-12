import serial
import time

# ================= 配置区 =================
SERIAL_PORT = '/dev/ttyACM0'  # 已改为你的端口
BAUDRATE = 1000000            # 飞特总线电机标准波特率
SERVO_ID = 6                  
# ==========================================

def read_reg(ser, sid, reg, size):
    """读取飞特伺服电机寄存器的通用底层函数"""
    # 构造指令包: [0xFF, 0xFF, ID, Length, INST, Address, Size, Checksum]
    msg = [sid, 0x04, 0x02, reg, size]
    checksum = ~(sum(msg)) & 0xFF
    packet = bytes([0xFF, 0xFF] + msg + [checksum])
    
    ser.reset_input_buffer()
    ser.write(packet)
    time.sleep(0.02) # 等待硬件响应
    
    if ser.in_waiting >= (6 + size):
        res = ser.read(ser.in_waiting)
        # 寻找包头 0xFF 0xFF
        try:
            idx = res.index(b'\xff\xff')
            data = res[idx+5 : idx+5+size]
            return int.from_bytes(data, byteorder='little')
        except: return None
    return None

def check_motor():
    print(f"🔍 正在连接 {SERIAL_PORT} 并清查 ID={SERVO_ID}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.5)
    except Exception as e:
        print(f"❌ 无法打开串口: {e}"); return

    # --- 1. 基础连通性 ---
    model = read_reg(ser, SERVO_ID, 0, 2)
    if model is None:
        print(f"❌ 错误：ID={SERVO_ID} 毫无响应！可能是线断了、ID不对或电源没接。")
        return
    print(f"✅ 连通性：正常 (电机型号代码: {model})")

    # --- 2. 扭矩状态 (关键) ---
    torque = read_reg(ser, SERVO_ID, 40, 1)
    print(f"📊 扭矩开关 (Torque Enable): {'【开启】' if torque == 1 else '【关闭】'}")
    if torque == 0:
        print("   👉 原因：电机处于放松状态。即使发指令它也不会转。")

    # --- 3. 硬件错误标志位 (核心排查) ---
    # 寄存器 69 是 Error 状态位
    error_code = read_reg(ser, SERVO_ID, 69, 1)
    if error_code is not None:
        print(f"🚨 硬件错误状态码: {error_code}")
        if error_code == 0:
            print("   ✅ 硬件无报错。")
        else:
            if error_code & (1 << 0): print("   ⚠️ 错误：电压异常 (Voltage Error)")
            if error_code & (1 << 1): print("   ⚠️ 错误：传感器异常 (Sensor Error)")
            if error_code & (1 << 2): print("   ⚠️ 错误：过温保护 (Temperature Error)")
            if error_code & (1 << 3): print("   ⚠️ 错误：堵转/过载保护 (Overload Error)")
    
    # --- 4. 运行模式 ---
    mode = read_reg(ser, SERVO_ID, 33, 1)
    print(f"⚙️ 工作模式: {'位置控制' if mode == 0 else '其它模式('+str(mode)+')'}")

    # --- 5. 实时数据对比 ---
    cur_pos = read_reg(ser, SERVO_ID, 56, 2)
    tgt_pos = read_reg(ser, SERVO_ID, 42, 2)
    load = read_reg(ser, SERVO_ID, 60, 2)
    print(f"📍 当前位置: {cur_pos} | 目标位置: {tgt_pos}")
    print(f"💪 当前负载 (Load): {load}")
    
    if torque == 1 and abs(cur_pos - tgt_pos) > 50:
        print("\n💡 诊断结果：扭矩已开且有位置偏差，但不转动。")
        if load > 50:
            print("   >>> 结论：机械卡死！电机在用力但转不动。请检查 6 号电机关节螺丝。")
        else:
            print("   >>> 结论：内部驱动电路或电机线圈可能损坏（空载且不转）。")
    elif torque == 0:
        print("\n💡 诊断结果：请在代码中检查是否执行了使能指令（Torque Enable = 1）。")

    ser.close()

if __name__ == "__main__":
    check_motor()