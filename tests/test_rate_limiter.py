from security.rate_limiter import (
    allow_request,
    key_for,
    lockout,
    remaining,
    reset
)


def test_allow():
    assert allow_request({'count': 2, 'start': 0}, 'ip', 5, 60, 1) is True

def test_block():
    assert allow_request({'count': 5, 'start': 0}, 'ip', 5, 60, 1) is False

def test_key_ip():
    assert key_for('1.2.3.4', None) == 'ip:1.2.3.4'

def test_key_user():
    assert key_for('1.2.3.4', 'u') == 'user:u'

def test_lockout():
    assert lockout(5, 5) is True

def test_remaining():
    assert remaining({'count': 2}, 5) == 3

def test_reset():
    assert reset({'count': 2}) == 0

def test_window():
    assert allow_request({'count': 5, 'start': 0}, 'ip', 5, 60, 61) is True

def test_empty():
    assert allow_request({'count': 0}, '', 5, 60, 1) is False

def test_burst():
    assert remaining({'count': 10}, 5) == 0

def test_success_reset():
    assert reset({'count': 3}) == 0

def test_unknown():
    assert key_for('', 'u') == 'user:u'

def test_different():
    assert key_for('1.1.1.1', 'u') != key_for('2.2.2.2', 'u')

def test_config():
    assert allow_request({'count': 1, 'start': 0}, 'ip', 2, 60, 1) is True

def test_bad():
    assert remaining({'count': -1}, 5) == 5
