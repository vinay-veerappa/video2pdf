
import browser_cookie3
import sys

try:
    print("Extracting tokens from Chrome...")
    cj = browser_cookie3.chrome(domain_name='.youtube.com')
    
    with open('cookies.txt', 'w') as f:
        f.write("# Netscape HTTP Cookie File\n")
        f.write("# http://curl.haxx.se/rfc/cookie.html\n\n")
        
        for cookie in cj:
            # Netscape format: domain, flag, path, secure, expiration, name, value
            flag = "TRUE" if cookie.domain.startswith('.') else "FALSE"
            path = cookie.path
            secure = "TRUE" if cookie.secure else "FALSE"
            expiration = str(int(cookie.expires)) if cookie.expires else "0"
            
            f.write(f"{cookie.domain}\t{flag}\t{path}\t{secure}\t{expiration}\t{cookie.name}\t{cookie.value}\n")
            
    print("Success! Saved to cookies.txt")
except Exception as e:
    print(f"Error: {e}")
