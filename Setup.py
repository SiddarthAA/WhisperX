import os
import time
import sys
import subprocess
from colorama import Fore, Style, init

init(autoreset=True)

def print_loading_animation():
    animation = "|/-\\"
    for i in range(10):
        sys.stdout.write(f"\r{Fore.CYAN}{Style.BRIGHT}Loading Resources!{animation[i % len(animation)]}")
        sys.stdout.flush()
        time.sleep(0.3)

def install_requirements():
    print(f"{Fore.RED}{Style.BRIGHT}\n\nInstalling dependencies from requirements.txt!")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print(f"{Fore.GREEN}{Style.BRIGHT}All dependencies have been successfully installed!")
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}{Style.BRIGHT}Error occurred while installing dependencies: {e}")

def create_directory(directory_name):
    current_directory = os.getcwd()
    full_path = os.path.join(current_directory, directory_name)
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Directory '{directory_name}' created successfully!")
    else:
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Directory '{directory_name}' already exists.")

def write_env_file(api_key):
    env_content = f"GEMINI_API_KEY={api_key}".strip()
    with open('.env', 'w') as env_file:
        env_file.write(env_content)
    print(f"{Fore.GREEN}{Style.BRIGHT}.env file created with the API key.")

def main():
    print("""

\n\n _       ____    _                     
| |     / / /_  (_)________  ___  _____
| | /| / / __ \/ / ___/ __ \/ _ \/ ___/
| |/ |/ / / / / (__  ) /_/ /  __/ /    
|__/|__/_/ /_/_/____/ .___/\___/_/     
                   /_/                 
\n""")
    print(f"{Fore.CYAN}{Style.BRIGHT}\nSetting up the environment...")
    print_loading_animation()

    install_requirements()
    
    new_directory = "Downloads"
    create_directory(new_directory)
    
    API = input(f"{Fore.RED}{Style.BRIGHT}\nEnter Gemini API KEY: ")
    write_env_file(API)
    
    print(f"\n\n{Fore.GREEN}{Style.BRIGHT}3/3 Task Completed! Setup Finishes! Run 'streamlit run App.py' To Get Started With Whisper!")

if __name__ == "__main__":
    main()