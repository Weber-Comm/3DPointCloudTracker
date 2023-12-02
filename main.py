import subprocess
import os

def run_script(script_name):
    """Run a Python script in a new Command Prompt window."""
    try:
        # Check if the script file exists
        if not os.path.exists(script_name):
            print(f"Error: {script_name} does not exist.")
            return

        # Command to open a new Command Prompt window
        cmd = f'start cmd.exe /k python "{script_name}"'

        # Start the Python script in a new Command Prompt window
        process = subprocess.Popen(cmd, shell=True)

        return process
    except Exception as e:
        print(f"Failed to run {script_name}: {e}")

if __name__ == "__main__":
    # Path to your Python scripts
    server_script = 'server.py'
    client_script = 'client.py'
    guest_script = 'guest.py'

    # Start each script in a new Command Prompt window
    server_process = run_script(server_script)
    client_process = run_script(client_script)
    guest_process = run_script(guest_script)

    # The scripts will continue to run in their respective windows