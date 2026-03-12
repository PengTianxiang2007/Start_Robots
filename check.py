import time
import serial

# ================= 配置参数 =================
SERIAL_PORT = '/dev/ttyACM0' 
BAUDRATE = 1000000 
# ===========================================

def read_register(sc, servo_id, reg_addr, length):
    # 构建 Feetech 读取寄存器的包 (INST_READ = 0x02)
    checksum = ~(servo_id + 0x04 + 0x02 + reg_addr + length) & 0xFF
    packet = bytearray([0xFF, 0xFF, servo_id, 0x04, 0x02, reg_addr, length, checksum])
    sc.write(packet)
    time.sleep(0.01)
    res = sc.read(sc.in_waiting)
    if len(res) >= 5 + length: # 0xFF 0xFF ID LEN ERR DATA... CHK
        return list(res[5:5+length])
    return None

def diagnostic():
    try:
        sc = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.05)
        print(f"--- 正在深入诊断 ID 1-6 的状态 ---")
        print(f"{'ID':<5} | {'扭矩使能':<8} | {'工作模式':<8} | {'当前位置':<8}")
        print("-" * 45)
    except Exception as e:
        print(f"无法打开串口: {e}"); return

    for i in range(1, 7):
        torque = read_register(sc, i, 40, 1) # Torque Enable
        mode = read_register(sc, i, 33, 1)   # Operating Mode
        pos = read_register(sc, i, 56, 2)    # Present Position (2 bytes)
        
        t_str = torque[0] if torque else "未知"
        m_str = mode[0] if mode else "未知"
        p_val = (pos[1] << 8) | pos[0] if pos else "未知"
        
        print(f"{i:<5} | {t_str:<12} | {m_str:<12} | {p_val:<12}")

    sc.close()

if __name__ == "__main__":
    diagnostic()