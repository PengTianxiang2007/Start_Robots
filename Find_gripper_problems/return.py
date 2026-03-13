import serial
import time

SERIAL_PORT = '/dev/ttyACM0'
BAUDRATE = 1000000
SERVO_ID = 6

def write_reg(ser, sid, reg, size, value):
    val_bytes = value.to_bytes(size, byteorder='little')
    msg = [sid, size + 3, 0x03, reg] + list(val_bytes)
    checksum = ~(sum(msg)) & 0xFF
    packet = bytes([0xFF, 0xFF] + msg + [checksum])
    ser.write(packet)
    time.sleep(0.05)

def fix_eeprom_limits():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.5)
    except Exception as e:
        print(f"❌ 串口打开失败: {e}"); return

    print(f"🛠️ 正在解除 ID={SERVO_ID} 的软件死锁...")

    # 1. 解锁 EEPROM (地址 48, 长度 1, 写入 0)
    write_reg(ser, SERVO_ID, 48, 1, 0)
    time.sleep(0.1)

    # 2. 写入正常的出厂宽容限位: Min=0, Max=4095
    print("🔓 写入新限位: Min -> 0, Max -> 4095")
    write_reg(ser, SERVO_ID, 9, 2, 0)      # Min Angle
    write_reg(ser, SERVO_ID, 11, 2, 4095)  # Max Angle
    time.sleep(0.1)

    # 3. 重新上锁 EEPROM (地址 48, 长度 1, 写入 1)
    write_reg(ser, SERVO_ID, 48, 1, 1)
    
    print("✅ 修复完毕！")
    ser.close()

if __name__ == "__main__":
    fix_eeprom_limits()