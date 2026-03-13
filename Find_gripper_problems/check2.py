import serial
import time

SERIAL_PORT = '/dev/ttyACM0'
BAUDRATE = 1000000
SERVO_ID = 6

def write_reg(ser, sid, reg, size, value):
    # 将数值拆分为字节 (小端序)
    val_bytes = value.to_bytes(size, byteorder='little')
    msg = [sid, size + 3, 0x03, reg] + list(val_bytes) # 0x03 是写指令
    checksum = ~(sum(msg)) & 0xFF
    packet = bytes([0xFF, 0xFF] + msg + [checksum])
    
    ser.reset_input_buffer()
    ser.write(packet)
    time.sleep(0.05)

def test_move():
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.5)
    
    print("1. 确保扭矩开启...")
    write_reg(ser, SERVO_ID, 40, 1, 1) # 地址 40, 长度 1, 值 1
    
    print("2. 下发新位置命令: 2243 (原位置 2043 + 200)...")
    write_reg(ser, SERVO_ID, 42, 2, 2243) # 地址 42(目标位置), 长度 2, 值 2243
    
    print("✅ 指令已发送！电机应该已经稍微转动了一下。")
    ser.close()

if __name__ == "__main__":
    test_move()