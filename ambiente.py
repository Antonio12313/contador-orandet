import time
import platform
import subprocess
import importlib.metadata

def coletar_ambiente():
    # ── Sistema operacional ───────────────────────────────────────────────────
    so = {
        "sistema":    platform.system(),
        "release":    platform.release(),
        "versao":     platform.version(),
        "maquina":    platform.machine(),
        "processador":platform.processor(),
        "python":     platform.python_version(),
    }

    # ── CPU ───────────────────────────────────────────────────────────────────
    cpu = {"nome": platform.processor() or "n/d", "nucleos_logicos": None, "nucleos_fisicos": None}
    try:
        import psutil
        cpu["nucleos_logicos"]  = psutil.cpu_count(logical=True)
        cpu["nucleos_fisicos"]  = psutil.cpu_count(logical=False)
        cpu["frequencia_max_MHz"] = round(psutil.cpu_freq().max, 1) if psutil.cpu_freq() else None
    except ImportError:
        # psutil não é obrigatório — tenta alternativas por SO
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo") as f:
                    linhas = f.read()
                nomes = [l.split(":")[1].strip() for l in linhas.splitlines() if "model name" in l]
                cpu["nome"] = nomes[0] if nomes else cpu["nome"]
            elif platform.system() == "Darwin":
                resultado = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True
                )
                cpu["nome"] = resultado.stdout.strip()
            elif platform.system() == "Windows":
                resultado = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True, text=True
                )
                linhas = [l.strip() for l in resultado.stdout.splitlines() if l.strip() and l.strip() != "Name"]
                cpu["nome"] = linhas[0] if linhas else cpu["nome"]
        except Exception:
            pass

    # ── RAM ───────────────────────────────────────────────────────────────────
    ram = {"total_GB": None, "disponivel_GB": None}
    try:
        import psutil
        mem = psutil.virtual_memory()
        ram["total_GB"]      = round(mem.total     / 1024 ** 3, 1)
        ram["disponivel_GB"] = round(mem.available / 1024 ** 3, 1)
    except ImportError:
        try:
            if platform.system() == "Linux":
                with open("/proc/meminfo") as f:
                    linhas = f.read()
                for linha in linhas.splitlines():
                    if "MemTotal" in linha:
                        ram["total_GB"] = round(int(linha.split()[1]) / 1024 ** 2, 1)
                    if "MemAvailable" in linha:
                        ram["disponivel_GB"] = round(int(linha.split()[1]) / 1024 ** 2, 1)
            elif platform.system() == "Darwin":
                resultado = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
                ram["total_GB"] = round(int(resultado.stdout.strip()) / 1024 ** 3, 1)
        except Exception:
            pass

    # ── GPU ───────────────────────────────────────────────────────────────────
    gpu = {"disponivel": False, "nome": "n/d", "nota": "pipeline CPU-only"}
    try:
        resultado = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if resultado.returncode == 0 and resultado.stdout.strip():
            gpu["disponivel"] = True
            gpu["nome"]       = resultado.stdout.strip().split(",")[0].strip()
            gpu["memoria"]    = resultado.stdout.strip().split(",")[1].strip()
    except Exception:
        pass

    # ── Versões de bibliotecas ────────────────────────────────────────────────
    libs = {}
    for pkg in ["numpy", "opencv-python", "opencv-python-headless",
                "scikit-learn", "scikit-image", "scipy",
                "xgboost", "lightgbm", "pandas", "joblib"]:
        try:
            libs[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            libs[pkg] = "não instalado"

    # opencv pode estar com nome diferente
    cv2_version = "não encontrado"
    for nome_cv in ["opencv-python", "opencv-python-headless", "opencv-contrib-python"]:
        if libs.get(nome_cv, "não instalado") != "não instalado":
            cv2_version = libs[nome_cv]
            break
    try:
        import cv2 as _cv2
        cv2_version = _cv2.__version__
    except Exception:
        pass
    libs["opencv_version_runtime"] = cv2_version

    return {
        "sistema_operacional": so,
        "cpu":                 cpu,
        "ram":                 ram,
        "gpu":                 gpu,
        "bibliotecas":         libs,
        "nota": (
            "Coletado automaticamente em tempo de execução. "
            "Para RAM disponível e núcleos de CPU, instalar psutil: pip install psutil"
        ),
    }