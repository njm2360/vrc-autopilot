def key_events():
    try:
        import msvcrt

        while True:
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # 特殊キーの先行バイトは読み捨て
                msvcrt.getwch()
                continue
            yield ch
    except ImportError:
        while True:
            line = input()
            yield line[:1] if line else " "
