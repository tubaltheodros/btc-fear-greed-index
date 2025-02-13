import schedule
import time
import subprocess

def run_script():
    try:
        subprocess.run(["/opt/anaconda3/bin/python3", "/Users/tubaltheodros/Desktop/btc-fear-greed-index/Bticoin_FG_Index.py"], check=True)
        print("Script ran successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")

# Schedule the task to run daily at 1 AM
schedule.every().day.at("01:00").do(run_script)

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(1)












