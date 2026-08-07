# utils/auth.py
def check_login(username, password):
    # You can later use DB or hashed passwords
    dummy_users = {
        "aathi": "1234",
        "guest": "guest"
    }
    return dummy_users.get(username) == password
