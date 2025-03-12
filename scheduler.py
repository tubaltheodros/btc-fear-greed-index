#!/usr/bin/env python3
import sys
import os

# Redirect standard streams at the very beginning
sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stdout.log'), 'a')
sys.stderr = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stderr.log'), 'a')

import schedule
import time
import subprocess
from pathlib import Path
import datetime
from datetime import datetime, timedelta

def run_btc_script():
    script_path = Path(__file__).parent / 'Bitcoin_FG_Index.py'
    print(f"Running BTC Fear & Greed Index script at {datetime.now()}")
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        print("Script completed successfully")
    except Exception as e:
        print(f"Error running script: {e}")

def check_missed_run():
    now = datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # If it's past midnight and before the next scheduled run
    if now > midnight and now < midnight + timedelta(hours=24):
        last_run_file = Path(__file__).parent / '.last_run'
        
        # Check if we already ran today
        if last_run_file.exists():
            with open(last_run_file, 'r') as f:
                last_run = datetime.fromisoformat(f.read().strip())
                if last_run.date() < now.date():
                    run_btc_script()
        else:
            run_btc_script()
            
        # Update last run time
        with open(last_run_file, 'w') as f:
            f.write(datetime.now().isoformat())

# Schedule for midnight
schedule.every().day.at("08:30").do(run_btc_script)

print(f"Scheduler started at {datetime.now()}")
print("Script will run at midnight every day or upon wake if midnight was missed")

# Check if we missed the midnight run when starting up
check_missed_run()

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(60)