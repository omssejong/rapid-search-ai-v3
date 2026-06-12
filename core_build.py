import subprocess
import os
import shutil
import glob

def build_core_as_module(output_dir=None):
    """
    Build the 'core' package as a .so module using Nuitka.
    """

    command = ["nuitka"]

    # --module: core 전체를 모듈화
    command += [
        "--module", "core",
        "--include-package=core",
        "--include-package=core.util",
        #"--include-package=core.Lee_utils",
        #"--include-package=core.models",
        #"--include-package=core.shin_utils",
        "--follow-import-to=core",
        "--follow-import-to=core.util",
        #"--follow-import-to=core.Lee_utils",
        #"--follow-import-to=core.models",
        #"--follow-import-to=core.shin_utils",
        "--remove-output",
        "--lto=yes",
        "--enable-plugin=no-qt,torch",
        "--disable-cache=all",
        "--clean-cache=all",
        "--jobs=0",
        "--assume-yes-for-downloads",
        "--show-progress",
        "--verbose",
        "--full-compat",
        "--module-parameter=torch-disable-jit=no"
    ]

    # 출력 경로 
    if output_dir:
        command.append(f"--output-dir={output_dir}")

    # 불필요한 파일
    exclude_files = [
        "images/*", "pt_models/*", "vid_folder/*", "image_test/*",
        "oms_models_send/*", "test/*", "venv/*", "test.json",
        "requirements.txt", "README.md", "nohup.out",
        ".git/*", ".gitignore", "build.py", "clip_bk/*"
    ]
    for file in exclude_files:
        command.append(f"--noinclude-data-files={file}")

    # 실행
    print(f"실행: {' '.join(command)}")
    subprocess.run(command, check=True)
    print("빌드 끝")

# 실제 실행
if __name__ == "__main__":
    root_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_path)

    build_core_as_module(output_dir=os.path.join(root_path, "build"))
