# SPDX-FileCopyrightText: 2017 Scott Shawcroft for Adafruit Industries
# SPDX-FileCopyrightText: 2018 Kattni Rembor for Adafruit Industries
# SPDX-FileCopyrightText: 2025 Craig Kolb
#
# SPDX-License-Identifier: MIT

"""
`SMBusServoKit`

Interface to a PCA9685 PWM controller attached to an I2C bus, in the style of adafruit-circuipython-servokit.
Uses smbus2 for communication.

* Author(s): Craig Kolb
"""

from smbus2 import SMBus
import time

__version__ = "0.0.1"
__repo__ = "https://github.com/cek/smbus-servokit.git"

_MODE1_REG = 0x00
_MODE2_REG = 0x01

# Mode1 bits
_ALLCALL_BIT = 0x01
_SLEEP_BIT = 0x10

# Mode2 bits
_OUTDRV_BIT = 0x04
_RESTART_BIT = 0x80

_PRESCALE_REG = 0xFE
_PCA9685_MAX_COUNT = 4096

# PWM registers are laid out sequentially, 4 per channel; 2 bytes to hold on and off times.
# LEDx_ON_L  = LED0_ON_L + 4 * channel
# LEDx_ON_H  = LEDx_ON_L + 1
# LEDx_OFF_L = LEDx_ON_L + 2
# LEDx_OFF_H = LEDx_ON_L + 3

_LED0_ON_L = 0x06

class ServoKit:
    """Class representing a PCA9685 PWM controller with attached servos.

    Uses smbus2 for communication, rather than, for example, busio.I2C.

    :param int channels: The number of servo channels available. Defaults to ``16``.
    :param int bus The index of I2C bus to use. Defaults to ``1``.
    :param int address: The address of the PCA9685 on the given I2C bus. Defaults to ``0x40``.
    :param int reference_clock_speed: The frequency of the internal clock in Hz. Defaults to ``25000000``.
    :param int frequency: The PWM frequency of the PCA9685 in Hz. Defaults to ``50``.

    """

    def __init__(self, *, channels: int = 16, bus: int = 1, address: int = 0x40, reference_clock_speed: int = 25000000, frequency: int = 50) -> None:
        if channels not in [8, 16]:
            raise ValueError('servo_channels must be 8 or 16.')
        self._items = [None] * channels
        self._servoFactory = _ServoFactory(self)
        self._channels = channels
        self._bus = SMBus(bus)
        self._address = address
        self._reference_clock_speed = reference_clock_speed
        self._cycle_length_us = None
        self._frequency = None
        self.frequency = frequency
        self.reset()

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, freq: int):
        if freq <= 0:
            raise ValueError('PCA9685 frequency must be positive')
        self._frequency = freq
        self._cycle_length_us = 1000000 / freq
        prescale_val = int(round(self._reference_clock_speed / float(_PCA9685_MAX_COUNT * freq)) - 1)

        # Read the mode1 register and toggle the sleep bit, which is required before setting the prescale value
        mode = self._i2cread(_MODE1_REG)
        mode = mode | _SLEEP_BIT
        self._i2cwrite(_MODE1_REG, mode)

        # Write the prescale value
        self._i2cwrite(_PRESCALE_REG, prescale_val)

        # Read the mode1 value
        sleep_mode = self._i2cread(_MODE1_REG)
        # Clear the sleep bit.
        mode = sleep_mode & ~_SLEEP_BIT
        self._i2cwrite(_MODE1_REG, mode)
        # If the restart bit was set while in sleep mode, wait for things to settle, and then restart the PWM output.
        if sleep_mode & _RESTART_BIT:
            time.sleep(0.001)
            mode = mode | _RESTART_BIT
            self._i2cwrite(_MODE1_REG, mode)

    def reset(self):
        """ Reset all servos. """
        self._i2cwrite(_MODE1_REG, _ALLCALL_BIT)
        self._i2cwrite(_MODE2_REG, _OUTDRV_BIT)
        time.sleep(0.001)


    def _i2cread(self, reg: int):
        return self._bus.read_byte_data(self._address, reg)

    def _i2cwrite(self, reg: int, val):
        self._bus.write_byte_data(self._address, reg, val)

    @property
    def servo(self) -> "_ServoFactory":
        return self._servoFactory

class Servo:
    """
    Basic servo class, in the style of adafruit_motor.servo. Used to control the position of a servo.

    :param ServoKit kit: Interface to controller to which the servo is attached.
    :param int channel: PCA9685 channel to which the controller is attached.
    :param float actuation_range: The physical range of motion of the sevo, in degrees, corresponding to ``min_pulse`` and ``max pulse`` values.
    :param int min_pulse: The minimum pulse width of the servo in microseconds.
    :param int max_pulse: The maximum pulse width of the servo in microseconds.
    """
    def __init__(self, kit: ServoKit, channel: int, actuation_range: float = 180., min_pulse: int = 750, max_pulse: int = 2250) -> None:
        self._kit = kit
        self._channel = channel
        if actuation_range <= 0:
            raise ValueError(f"servo actuation range must be greater than 0")
        self.actuation_range = actuation_range
        self.set_pulse_width_range(min_pulse, max_pulse)
        self._angle = None

    def set_pulse_width_range(self, min_pulse: int, max_pulse: int) -> None:
        """Set the minimum (angle=0) and maximum (angle=actuation range) pulse width values."""
        if min_pulse < 0 or max_pulse < 0:
            raise ValueError(f'Pulse widths must be positive.')
        if max_pulse <= min_pulse:
            raise ValueError(f'Max pulse width must be greater than min pulse width.')
        self.min_pulse = min_pulse
        self.max_pulse = max_pulse

    def __len__(self) -> int:
        return len(self._kit._items)

    @property
    def fraction(self) -> float:
        """Servo angle as a faction of the actuation range."""
        return self._angle / self.actuation_range

    @fraction.setter
    def fraction(self, fraction: float) -> None:
        self.angle = fraction * self.actuation_range

    @property
    def angle(self):
        """Servo angle, in degrees."""
        return self._angle

    @angle.setter
    def angle(self, angle):
        if angle is None:
            return
        if angle < 0 or angle > self.actuation_range:
            raise ValueError(f"Angle must be between 0 and {self.actuation_range}, rather than {angle}.")

        self._angle = angle

        pulse_width = self.min_pulse + (angle / float(self.actuation_range)) * (self.max_pulse - self.min_pulse)
        pwm_value = int((pulse_width / self._kit._cycle_length_us) * _PCA9685_MAX_COUNT)

        if pwm_value < 1:
            pwm_value = 1

        reg_base = _LED0_ON_L + 4 * self._channel
        # LEDx_ON_L and _H = 0
        self._kit._i2cwrite(reg_base, 0)
        self._kit._i2cwrite(reg_base + 1, 0)
        # Write pwm_value to LEDx_OFF_L and _H
        self._kit._i2cwrite(reg_base + 2, pwm_value & 0xFF)
        self._kit._i2cwrite(reg_base + 3, pwm_value >> 8)

class _ServoFactory:
    def __init__(self, kit: ServoKit):
        self._kit = kit
    def __getitem__(self, servo_channel: int) -> Servo:
        if servo_channel >= self._kit._channels or servo_channel < 0:
            raise ValueError(f"servo index {servo_channel} is out of range (0-{self._kit_channels - 1}).")
        servo = self._kit._items[servo_channel]
        if servo is None:
            servo = Servo(self._kit, servo_channel)
            self._kit._items[servo_channel] = servo
        return servo

    def __len__(self) -> int:
        return len(self._kit._items)
