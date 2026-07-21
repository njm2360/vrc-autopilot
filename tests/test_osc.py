"""VRChatOSC の送信テスト(VRChat不要、ループバックUDPで受信して python-osc で解析)。"""

from __future__ import annotations

import socket

import pytest
from pythonosc.osc_message import OscMessage

from vrc_autopilot.control.osc import VRChatOSC
from vrc_autopilot.perception.spec import HUD_ENABLE_PARAM


def _receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(1.0)
    return sock, sock.getsockname()[1]


def _recv(sock):
    data, _ = sock.recvfrom(2048)
    msg = OscMessage(data)
    return msg.address, list(msg.params)


def test_move_sends_clamped_axes():
    sock, port = _receiver()
    try:
        osc = VRChatOSC("127.0.0.1", port)
        osc.move(forward=2.0, strafe=-3.0)
        a1, p1 = _recv(sock)
        a2, p2 = _recv(sock)
        assert a1 == "/input/Vertical" and p1[0] == pytest.approx(1.0)
        assert a2 == "/input/Horizontal" and p2[0] == pytest.approx(-1.0)
    finally:
        sock.close()


def test_look_turn_and_pitch():
    sock, port = _receiver()
    try:
        osc = VRChatOSC("127.0.0.1", port)
        osc.look(0.3, pitch=0.5)
        a1, p1 = _recv(sock)
        a2, p2 = _recv(sock)
        assert a1 == "/input/LookHorizontal" and p1[0] == pytest.approx(0.3)
        assert a2 == "/input/LookVertical" and p2[0] == pytest.approx(0.5)
    finally:
        sock.close()


def test_look_without_pitch_zeroes_vertical():
    # /input 軸は最後の値を保持するため、pitch=0 も明示送信して止める必要がある
    sock, port = _receiver()
    try:
        osc = VRChatOSC("127.0.0.1", port)
        osc.look(0.2)
        a1, p1 = _recv(sock)
        a2, p2 = _recv(sock)
        assert a1 == "/input/LookHorizontal" and p1[0] == pytest.approx(0.2)
        assert a2 == "/input/LookVertical" and p2[0] == pytest.approx(0.0)
    finally:
        sock.close()


def test_button_is_int():
    sock, port = _receiver()
    try:
        VRChatOSC("127.0.0.1", port).button("Jump", True)
        a, p = _recv(sock)
        assert a == "/input/Jump"
        assert p == [1] and isinstance(p[0], int)
    finally:
        sock.close()


def test_avatar_param():
    sock, port = _receiver()
    try:
        VRChatOSC("127.0.0.1", port).avatar_param(HUD_ENABLE_PARAM, True)
        a, p = _recv(sock)
        assert a == f"/avatar/parameters/{HUD_ENABLE_PARAM}"
        assert p == [True]
    finally:
        sock.close()


def test_stop_zeroes_axes():
    sock, port = _receiver()
    try:
        osc = VRChatOSC("127.0.0.1", port)
        osc.stop()
        seen = {}
        for _ in range(4):
            a, p = _recv(sock)
            seen[a] = p[0]
        assert seen["/input/Vertical"] == pytest.approx(0.0)
        assert seen["/input/Horizontal"] == pytest.approx(0.0)
        assert seen["/input/LookHorizontal"] == pytest.approx(0.0)
        assert seen["/input/LookVertical"] == pytest.approx(0.0)
    finally:
        sock.close()
