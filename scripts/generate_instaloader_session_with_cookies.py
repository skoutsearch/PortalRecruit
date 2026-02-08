import instaloader
import http.cookiejar

# Your info
USERNAME = 'portalrecruit'
COOKIE_FILE = 'cookies.txt'  # Ensure this file is in the same folder

L = instaloader.Instaloader()

# Load cookies using the standard library
cj = http.cookiejar.MozillaCookieJar(COOKIE_FILE)
try:
    cj.load(ignore_discard=True, ignore_expires=True)
except FileNotFoundError:
    print(f"Error: Can't find {COOKIE_FILE}. Make sure it's in this folder.")
    exit()

# Inject into Instaloader
L.context._session.cookies.update(cj)

# Test and save
try:
    # We set the username manually so Instaloader knows who the session belongs to
    L.context.username = USERNAME
    L.test_login()
    L.save_session_to_file()
    print(f"✅ Success! Session saved for {USERNAME}")
except Exception as e:
    print(f"❌ Failed: {e}")
