import os
import subprocess

def run_executables():
    # 获取当前工作目录
    current_directory = os.getcwd()

    # 获取当前目录下的所有文件
    files = [f for f in os.listdir(current_directory) if os.path.isfile(f) and os.access(f, os.X_OK)]

    # 打开一个文件用于保存结果
    with open("execution_results.txt", "w") as result_file:
        # 运行每个可执行文件6次
        for file in files:
            for run_count in range(6):
                print(f"Running {file} ({run_count+1}/6)")
                command = f"./{file}"
                
                try:
                    result = subprocess.run(command, shell=True, check=True, capture_output=True)
                    result_output = result.stdout.decode('utf-8')
                    result_file.write(f"File: {file}, Run: {run_count+1}\n")
                    result_file.write(f"Output:\n{result_output}\n")
                except subprocess.CalledProcessError as e:
                    result_file.write(f"Error running {file}, Run: {run_count+1}\n")
                    result_file.write(f"Error Message: {e}\n")

if __name__ == "__main__":
    run_executables()

