import os

os.environ["DEBUG"] = "False"

def debug(msg):        
    if os.environ["DEBUG"] == "True":
        print("DEBUG:" + msg)
