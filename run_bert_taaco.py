import subprocess

def run_bert():
    print("Running BERT.py...")
    subprocess.run(['python', 'BERT.py'])  # BERT.py 실행
    print("BERT.py finished.")

def run_taaco():
    print("Running TAACO.py...")
    subprocess.run(['python', 'TAACO.py'])  # TAACO.py 실행
    print("TAACO.py finished.")

if __name__ == "__main__":
    run_bert()  # BERT.py 실행
    run_taaco()  # BERT.py 실행 후 TAACO.py 실행
