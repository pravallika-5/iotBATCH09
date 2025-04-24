#program for dc motor:
import RPi.GPIO as GPIO
import time

IN1 = 7
IN2 = 8
EN1 = 12

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(EN1, GPIO.OUT)

GPIO.output(EN1, GPIO.HIGH)

def motor_forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)

def stop_motor():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(EN1, GPIO.LOW)
    print("Motor stopped")

motor_forward()

try:
    print("Press Ctrl+C to stop the motor manually...")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Manually stopped.")
    stop_motor()
    GPIO.cleanup()
